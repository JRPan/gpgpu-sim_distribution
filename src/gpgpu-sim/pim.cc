#include "pim.h"
#include <float.h>
#include <limits.h>
#include <string.h>
#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_sim.h"
#include "../statwrapper.h"
#include "addrdec.h"
#include "dram.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "icnt_wrapper.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "pim_icnt_wrapper.h"
#include "shader.h"
#include "shader_trace.h"
#include "stat-tool.h"
#include "traffic_breakdown.h"
#include "visualizer.h"

#include <functional>
#include <unordered_set>

pim_core_ctx::pim_core_ctx(class gpgpu_sim *gpu,
                           class pim_core_cluster *cluster, unsigned shader_id,
                           unsigned tpc_id, const shader_core_config *config,
                           const memory_config *mem_config,
                           shader_core_stats *stats,
                           pim_core_config *pim_config,
                           pim_core_stats *pim_stats)
    : core_t(gpu, NULL, config->warp_size, config->n_thread_per_shader) {
  m_sid = shader_id;
  m_tpc = tpc_id;
  m_stats = stats;
  m_cluster = cluster;
  m_config = config;
  m_memory_config = mem_config;
  m_icnt = new pim_memory_interface(this, m_cluster);
  m_mem_fetch_allocator =
      new shader_core_mem_fetch_allocator(m_sid, m_tpc, m_memory_config);
  m_pim_core_config = pim_config;
  exec_shader_core_ctx *dummy_shader = new exec_shader_core_ctx(
      gpu, NULL, -1, -1, m_config, m_memory_config, m_stats);
  dummy_shader->resize_shd_warp(m_pim_core_config->num_xbars);
  m_scoreboard = new Scoreboard(m_sid, m_pim_core_config->num_xbars, gpu);
  m_ldst_unit = new simple_ldst_unit(m_icnt, m_mem_fetch_allocator, m_config,
                                     m_memory_config, m_stats, m_sid, m_tpc,
                                     gpu, dummy_shader, this, m_scoreboard);
  m_pim_stats = pim_stats;

  sent_bytes = 0;
  used_xbars = 0;

  m_pending_loads = 0;
  last_checked_xbar = 0;

  assert(m_pim_core_config->byte_per_row() % mf_size == 0);
}

void pim_core_ctx::warp_exit(unsigned warp_id) {}

bool pim_core_ctx::warp_waiting_at_barrier(unsigned warp_id) const {
  return false;
}

// cluster

void pim_core_ctx::issue_warp(register_set &pipe_reg_set,
                              const warp_inst_t *next_inst,
                              const active_mask_t &active_mask,
                              unsigned warp_id, unsigned sch_id) {}
void pim_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                              unsigned end_thread, unsigned ctaid, int cta_size,
                              kernel_info_t &kernel) {}

void pim_core_ctx::cycle() {
  commit();
  execute();
  issue();
}

void pim_core_ctx::execute() {
  unsigned cycle = get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  // for (unsigned i = 0; i < num_result_bus; i++) {
  //   *(m_result_bus[i]) >>= 1;
  // }
  for (unsigned n = 0; n < m_pim_core_config->num_xbars; n++) {
    if (!m_xbars[n]->active) continue;
    unsigned multiplier = m_xbars[n]->clock_multiplier();
    for (unsigned c = 0; c < multiplier; c++) {
      m_xbars[n]->cycle();
    }
    /*
    register_set *issue_inst = m_issue_reg[n];
    warp_inst_t **ready_reg = issue_inst->get_ready();
    if (issue_inst->has_ready() && m_xbars[n]->can_issue(**ready_reg)) {
      assert((*ready_reg)->latency < MAX_ALU_LATENCY);
      bool schedule_wb_now = !m_xbars[n]->stallable();
      int resbus = -1;
      m_xbars[n]->issue(*issue_inst);
      // if (schedule_wb_now &&
      //     (resbus = test_res_bus((*ready_reg)->latency)) != -1) {
      //   assert((*ready_reg)->latency < MAX_ALU_LATENCY);
      //   m_result_bus[resbus]->set((*ready_reg)->latency);
      //   printf("xbar %u issue %u, %u\n", n, (*ready_reg)->op, cycle);
      //   m_xbars[n]->issue(*issue_inst);
      // } else if (!schedule_wb_now) {
      //   printf("xbar %u issue %u, %u\n", n, (*ready_reg)->op, cycle);
      //   m_xbars[n]->issue(*issue_inst);
      // } else {
      //   // stall issue (cannot reserve result bus)
      // }
      // m_xbars[n]->issue(*issue_inst);
    }
    */
  }

  // m_ldst_unit->cycle();
  register_set *issue_inst = m_ldst_reg[0];
  warp_inst_t **ready_reg = issue_inst->get_ready();
  if (issue_inst->has_ready() && m_ldst_unit->can_issue(**ready_reg)) {
    m_ldst_unit->issue(*issue_inst);
  }
}

void pim_core_ctx::control_cycle() {
  unsigned cycle = get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  unsigned checked_xbars = 0;

  // unsigned xbar_id = last_checked_xbar;
  // for (unsigned n = 0; n < m_pim_core_config->num_xbars; n++) {
  //   assert(xbar_id < m_pim_core_config->num_xbars);
  //   pim_xbar *xbar = m_xbars[xbar_id];
  //   xbar_id = (xbar_id + 1) % m_pim_core_config->num_xbars;
  for (auto xbar : m_xbars) {
    if (!xbar->active) continue;
    checked_xbars++;
    m_pim_stats->xbar_tot_cycle[get_pim_cluster_id()][xbar->m_xbar_id]++;

    if (xbar->m_status == XBAR_INITIATED) {
      /*
      unsigned size = 32;
      for (unsigned row = 0; row < xbar->used_rows; row++) {
        unsigned byte_per_row = xbar->byte_per_row;
        new_addr_type row_addr = xbar->weight.addr + row * byte_per_row;
        unsigned sent_bytes = 0;
        while (sent_bytes < byte_per_row) {
          new_addr_type addr = row_addr + sent_bytes;
          // to sector
          addr =
              get_gpu()->getShaderCoreConfig()->m_L1D_config.sector_addr(addr);
          warp_inst_t *inst = new warp_inst_t(m_config);
          inst->op = LOAD_OP;
          inst->cache_op = CACHE_ALL;
          inst->data_size = 4;
          inst->set_addr(0, addr);
          inst->space.set_type(global_space);
          inst->pim_xbar_id = xbar->m_xbar_id;
          // inst->incount = 0;
          inst->outcount = 1;
          inst->out[0] = 1;
          xbar->inst_queue.push(inst);

          sent_bytes += size;
        }
        warp_inst_t *inst = new warp_inst_t(m_config);
        inst->op = XBAR_PROGRAM_OP;
        inst->latency = m_pim_core_config->program_latency;
        inst->pim_xbar_id = xbar->m_xbar_id;
        inst->incount = 1;
        // inst->outcount = 0;
        inst->in[0] = 1;
        xbar->inst_queue.push(inst);
      }
      xbar->m_status = XBAR_PROGRAM;
      */
    } else if (xbar->m_status == XBAR_PROGRAM) {
    } else if (xbar->m_status == XBAR_PROGRAMMED) {
      printf("core %u xbar %u start compute, %u\n", m_tpc, xbar->m_xbar_id,
             cycle);
      xbar->m_status = XBAR_COMPUTING;
    } else if (xbar->m_status == XBAR_COMPUTING) {
      new_addr_type size = m_config->m_L1D_config.get_line_sz();
      unsigned input_repeat = m_pim_core_config->device_precision /
                              m_pim_core_config->dac_precision;

      unsigned counter = 0;
      while (xbar->op_id < xbar->m_layer->input_size) {
        // while (xbar->op_id < xbar->m_layer->input_size &&
        // xbar->inst_queue.size() < 100) {

        std::set<new_addr_type> addrs = gen_addr(xbar);
        if (addrs.size() != 0) {
          std::vector<new_addr_type> addr_needed;
          for (auto addr : addrs) {
            addr = addr & ~(size - 1);
            if (xbar->regs_value.find(addr) == xbar->regs_value.end()) {
              if (xbar->regs_order.size() > m_pim_core_config->num_input_regs) {
                xbar->regs_value.erase(xbar->regs_order.front());
                xbar->regs_order.pop_front();
              }
              xbar->regs_order.push_back(addr);
              xbar->regs_value.insert(addr);

              addr_needed.push_back(addr);
              if (addr_needed.size() == m_config->m_L1D_config.l1_banks) {
                warp_inst_t *inst = new warp_inst_t(m_config);
                inst->op = PSEUDO_LD_OP;
                inst->outcount = 1;
                inst->out[0] = reg;
                xbar->inst_queue.push(inst);
                m_addr_map.insert(std::make_pair(inst, addr_needed));
                addr_needed.clear();
              }
            }
          }
          if (addr_needed.size() != 0) {
            warp_inst_t *inst = new warp_inst_t(m_config);
            inst->op = PSEUDO_LD_OP;
            inst->outcount = 1;
            inst->out[0] = reg;

            xbar->inst_queue.push(inst);
            m_addr_map.insert(std::make_pair(inst, addr_needed));
          }

          assert(xbar->inst_queue.back()->op == PSEUDO_LD_OP);
          m_pivot_load.insert(xbar->inst_queue.back());

          for (unsigned k = 0; k < input_repeat; k++) {
            warp_inst_t *inst = new warp_inst_t(m_config);
            inst->op = XBAR_INTEGRATE_OP;
            inst->latency = m_pim_core_config->integrate_latency;
            inst->incount = 1;
            inst->outcount = 1;
            inst->pim_xbar_id = xbar->m_xbar_id;
            inst->in[0] = reg;       // load result
            inst->out[0] = reg + 1;  // integrate result
            inst->ar1 = 0;
            inst->ar2 = 0;
            xbar->inst_queue.push(inst);
            counter++;

            // add sample op
            inst = new warp_inst_t(m_config);
            inst->op = XBAR_SAMPLE_OP;
            inst->latency = m_pim_core_config->sample_latency;
            inst->incount = 1;
            inst->outcount = 1;
            inst->pim_xbar_id = xbar->m_xbar_id;
            inst->in[0] = reg + 1;   // integrate result
            inst->out[0] = reg + 2;  // sample result
            inst->ar1 = 0;
            inst->ar2 = 0;
            xbar->inst_queue.push(inst);
          }
        }

        // printf("xbar %u compiling op %u\n", xbar->m_xbar_id, xbar->op_id);
        unsigned total_rows = xbar->m_layer->global_height;
        unsigned pos_x = xbar->op_id % total_rows;
        unsigned pos_y = xbar->op_id / total_rows;

        unsigned split_at = (xbar->xbar_row_id + 1) * xbar->used_rows;
        if (split_at > total_rows) {
          split_at = total_rows;
        }

        if (pos_x + m_pim_core_config->row_activation_rate >= split_at) {
          xbar->op_id =
              (pos_y + 1) * total_rows + xbar->xbar_row_id * xbar->used_rows;
        } else {
          xbar->op_id += m_pim_core_config->row_activation_rate;
        }

        const bool enable_write = false;
        if ((xbar->op_id) % (xbar->used_rows) == 0 && enable_write) {
          assert(xbar->inst_queue.back()->op == XBAR_SAMPLE_OP);
          warp_inst_t *inst = new warp_inst_t(m_config);
          unsigned offset = (xbar->op_id % xbar->used_rows) * xbar->used_cols *
                            m_pim_core_config->get_data_size_byte();
          // TODO: correct offset?
          new_addr_type addr = xbar->m_layer->output_addr + offset;
          inst->op = STORE_OP;
          inst->cache_op = CACHE_ALL;
          inst->data_size = 1;
          inst->set_addr(0, addr);
          inst->space.set_type(global_space);
          inst->pim_xbar_id = xbar->m_xbar_id;
          inst->incount = 1;
          inst->in[0] = xbar->inst_queue.back()->out[0];
          xbar->inst_queue.push(inst);
        }

        if (xbar->op_id >= xbar->m_layer->input_size) {
          // if (xbar->op_id >= 200) {
          // last op is exit
          warp_inst_t *inst = new warp_inst_t(m_config);
          inst->op = EXIT_OPS;
          inst->latency = 0;
          xbar->inst_queue.push(inst);

          xbar->op_id = xbar->m_layer->input_size;
        }
      }

      counter++;  // gdb breakpoint - not really adding 1
    } else if (xbar->m_status == XBAR_DONE) {
      std::vector<pim_xbar *> layer_xbars = xbar->m_layer->m_xbars;
      bool all_done = true;
      for (auto xbar : layer_xbars) {
        if (xbar->m_status != XBAR_DONE) {
          all_done = false;
          break;
        }
      }

      if (all_done) {
        m_gpu->m_finished_layers.insert(xbar->m_layer);
        for (auto c_bar : layer_xbars) {
          assert(c_bar->inst_queue.size() == 0);

          c_bar->active = false;
        }
      }
    }
  }
  // last_checked_xbar = (last_checked_xbar + 1) % m_pim_core_config->num_xbars;
}

void pim_core_ctx::issue() {
  unsigned cycle = get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  unsigned xbar_id = last_checked_xbar;
  std::set<new_addr_type> addrs_to_issue;
  std::vector<unsigned> merged_xbars;
  bool ld_full = false;
  for (unsigned n = 0; n < m_pim_core_config->num_xbars; n++) {
    assert(xbar_id < m_pim_core_config->num_xbars);
    pim_xbar *xbar = m_xbars[xbar_id];
    xbar_id = (xbar_id + 1) % m_pim_core_config->num_xbars;
    if (!xbar->active) continue;

    // inst queue -> pipeline register
    if (xbar->inst_queue.size() > 0) {
      bool issued = false;
      warp_inst_t *inst = xbar->inst_queue.front();
      switch (inst->op) {
        case PSEUDO_LD_OP: {  // load but addrs need to be merged
          assert(m_scoreboard->checkCollision(xbar->m_xbar_id, inst) == false);
          std::set<new_addr_type> pending_addrs;
          if (m_ldst_reg[0]->has_free()) {
            for (auto addr : m_addr_map[inst]) {
              if (addrs_to_issue.find(addr) == addrs_to_issue.end()) {
                pending_addrs.insert(addr);
              }
            }
            if (addrs_to_issue.size() + pending_addrs.size() <=
                m_config->m_L1D_config.l1_banks) {
              merged_xbars.push_back(xbar->m_xbar_id);
              for (auto addr : pending_addrs) {
                assert(addrs_to_issue.find(addr) == addrs_to_issue.end());
                addrs_to_issue.insert(addr);
              }
              inst->issue(
                  active_mask_t().set(), xbar->m_xbar_id,
                  get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle, 0,
                  0);
              assert(inst->outcount > 0);
              if (m_pivot_load.find(inst) != m_pivot_load.end()) {
                m_scoreboard->reserveRegisters(inst);
                m_pivot_load.erase(inst);
              }
              m_addr_map.erase(inst);
              issued = true;
            } else {
              ld_full = true;
            }
          }
          break;
        }
        case STORE_OP:
          if (m_scoreboard->checkCollision(xbar->m_xbar_id, inst) == false) {
            if (addrs_to_issue.empty()) {
              issued = issue_mem(inst, active_mask_t().set(0));
              if (issued) {
                m_scoreboard->reserveRegisters(inst);
              }
            } else {
              m_pim_stats->xbar_stall[XBAR_STALL_LD_ST_CONFLICT]++;
            }
          } else {
          }
          break;
        case XBAR_INTEGRATE_OP:
        case XBAR_PROGRAM_OP:
        case XBAR_SAMPLE_OP:
        case EXIT_OPS:
          if (m_scoreboard->checkCollision(xbar->m_xbar_id, inst) == false) {
            if (m_issue_reg[xbar->m_xbar_id]->has_free()) {
              warp_inst_t **pipe_reg = m_issue_reg[xbar->m_xbar_id]->get_free();
              assert(pipe_reg);
              **pipe_reg = *inst;
              (*pipe_reg)->issue(
                  active_mask_t().set(0), inst->pim_xbar_id,
                  get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle, -1,
                  -1);
              m_scoreboard->reserveRegisters(*pipe_reg);
              issued = true;
            }
          } else {
          }
          break;
        default:
          assert(0 && "Invalid xbar op");
      }

      if (issued) {
        delete inst;
        inst = NULL;
        xbar->inst_queue.pop();
      }
    }

    // pipeline register -> xbar
    register_set *issue_inst = m_issue_reg[xbar->m_xbar_id];
    warp_inst_t **ready_reg = issue_inst->get_ready();
    if (issue_inst->has_ready() && xbar->can_issue(**ready_reg)) {
      assert((*ready_reg)->latency < MAX_ALU_LATENCY);
      xbar->issue(*issue_inst);
    }
  }
  if (addrs_to_issue.size() != 0) {
    active_mask_t active_mask;
    warp_inst_t *inst = new warp_inst_t(m_config);
    inst->op = LOAD_OP;
    inst->cache_op = CACHE_ALL;
    inst->data_size = 1;
    inst->space.set_type(global_space);
    inst->pim_xbar_id = -1;
    // inst->incount = 0;
    inst->outcount = 1;
    inst->out[0] = 1;
    inst->ar1 = 0;
    inst->ar2 = 0;

    unsigned index = 0;
    // printf("load addr ");
    for (auto addr : addrs_to_issue) {
      // printf("%llx ", addr);
      inst->set_addr(index, addr);
      active_mask.set(index);
      index++;
    }
    // printf("\n");

    inst->m_multicast_xbars = merged_xbars;
    bool issued_mem = issue_mem(inst, active_mask);
    assert(issued_mem);
    delete inst;
  }

  if (ld_full) {
    m_pim_stats->xbar_stall[XBAR_STALL_LD_FULL]++;
  }

  last_checked_xbar = (last_checked_xbar + 1) % m_pim_core_config->num_xbars;
}

void pim_core_ctx::commit() {
  unsigned cycle = get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  for (unsigned n = 0; n < m_pim_core_config->num_xbars; n++) {
    if (!m_xbars[n]->active) continue;
    if (m_result_reg[n]->has_ready()) {
      warp_inst_t **ready_reg = m_result_reg[n]->get_ready();
      m_scoreboard->releaseRegisters(*ready_reg);
      (*ready_reg)->clear();
      m_pim_stats->xbar_executed_inst[m_tpc - m_config->n_simt_clusters][n]++;

      if ((*ready_reg)->op == XBAR_PROGRAM_OP) {
        // printf("xbar %u finished programming %u, %u\n", n,
        //        m_xbars[n]->programmed_rows, cycle);

        m_xbars[n]->programmed_rows++;
        if (m_xbars[n]->programmed_rows == m_xbars[n]->used_rows) {
          // printf("xbar %u done programming, %u\n", n, cycle);
          m_xbars[n]->m_status = XBAR_PROGRAMMED;
        }
      } else if ((*ready_reg)->op == XBAR_INTEGRATE_OP) {
        // if (n == 0 && m_tpc == 80)
        // printf("xbar %u done computing, %u, \n", n, cycle);
      } else if ((*ready_reg)->op == XBAR_SAMPLE_OP) {
        m_xbars[n]->done_activation++;
        // if (n == 0 && m_tpc == 80)
        // printf("xbar %u done sampling, %u, %u\n", n, cycle,
        //  m_xbars[n]->done_activation);
        if (m_xbars[n]->done_activation % 10000 == 0) {
          printf("core %u xbar %u done sampling, %u, %u\n", m_tpc, n,
                 m_xbars[n]->done_activation, cycle);
          fflush(stdout);
        }
      } else if ((*ready_reg)->op == EXIT_OPS) {
        printf("core %u xbar %u done layer, %u\n", m_tpc, n, cycle);
        m_xbars[n]->m_status = XBAR_DONE;
      }
    }
  }
}

bool pim_core_ctx::response_buffer_full() const {
  return m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size;
}

void pim_core_ctx::accept_response(mem_fetch *mf) { m_ldst_unit->fill(mf); }

void pim_core_ctx::ldst_cycle() { m_ldst_unit->cycle(); }

std::set<new_addr_type> pim_core_ctx::gen_addr(pim_xbar *xbar) {
  std::set<new_addr_type> addrs;
  for (unsigned j = 0; j < m_pim_core_config->row_activation_rate; j++) {
    new_addr_type addr = xbar->m_layer->matmul_addr[xbar->op_id + j];
    // addr = addr & ~3; // align to 4B
    if ((xbar->op_id + j) == xbar->m_layer->input_size) {
      break;  // out of bound
    }
    if (addr == 0) {
      continue;
    }
    if (addrs.find(addr) == addrs.end()) {
      addrs.insert(addr);
    }
  }

  return addrs;
}

void pim_core_cluster::ldst_cycle() {
  for (unsigned i = 0; i < m_config->n_pim_cores_per_cluster; i++)
    m_core[i]->ldst_cycle();
}

void pim_core_cluster::core_cycle() {
  for (unsigned i = 0; i < m_config->n_pim_cores_per_cluster; i++)
    m_core[i]->cycle();
}

void pim_core_cluster::core_control_cycle() {
  for (unsigned i = 0; i < m_config->n_pim_cores_per_cluster; i++)
    m_core[i]->control_cycle();
}

void pim_core_cluster::get_cache_stats(cache_stats &cs) const {
  for (unsigned i = 0; i < m_config->n_pim_cores_per_cluster; ++i) {
    m_core[i]->get_cache_stats(cs);
  }
}

void pim_core_cluster::icnt_cycle() {
  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front();
    unsigned cid = m_config->sid_to_cid(mf->get_sid());
    if (!m_core[cid]->response_buffer_full()) {
      m_response_fifo.pop_front();
      m_memory_stats->memlatstat_read_done(mf);
      m_core[cid]->accept_response(mf);
    }
  }
  if (m_response_fifo.size() < m_config->n_simt_ejection_buffer_size) {
    mem_fetch *mf = (mem_fetch *)::icnt_pop(m_cluster_id);
    if (!mf) return;
    assert(mf->get_tpc() == m_cluster_id);
    assert(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK);

    // The packet size varies depending on the type of request:
    // - For read request and atomic request, the packet contains the data
    // - For write-ack, the packet only has control metadata
    unsigned int packet_size =
        (mf->get_is_write()) ? mf->get_ctrl_size() : mf->size();
    m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size);
    mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,
                   get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle);
    m_response_fifo.push_back(mf);
    m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
  }
}

unsigned pim_core_config::get_data_size_byte() {
  switch (data_type) {
    case INT8_TYPE:
      return 1;
    case FP16_TYPE:
    case INT16_TYPE:
      return 2;
    case FP32_TYPE:
    case INT32_TYPE:
      return 4;
    case FP64_TYPE:
    case INT64_TYPE:
      return 8;
    default:
      assert(0 && "Invalid data type");
      return 0;
  }
}

unsigned pim_core_config::get_data_size_bit() {
  switch (data_type) {
    case INT8_TYPE:
      return 8;
    case FP16_TYPE:
    case INT16_TYPE:
      return 16;
    case FP32_TYPE:
    case INT32_TYPE:
      return 32;
    case FP64_TYPE:
    case INT64_TYPE:
      return 64;
    default:
      assert(0 && "Invalid data type");
      return 0;
  }
}

unsigned pim_core_config::byte_per_row() {
  return get_data_size_byte() * xbar_size_x * num_device_per_weight();
}

unsigned pim_core_config::num_device_per_weight() {
  unsigned devices = get_data_size_bit() / device_precision;
  assert(devices > 0);
  return devices;
}

int pim_core_ctx::test_res_bus(int latency) {
  for (unsigned i = 0; i < num_result_bus; i++) {
    if (!m_result_bus[i]->test(latency)) {
      return i;
    }
  }
  return -1;
}

void pim_xbar::active_lanes_in_pipeline() {}

void pim_xbar::issue(register_set &source_reg) {
  warp_inst_t **ready_reg = source_reg.get_ready();
  source_reg.move_out_to(m_dispatch_reg);
  // simd_function_unit::issue(source_reg);
}

// bool pim_xbar::xbar_icnt_injection_buffer_full(unsigned size, bool write) {
//   unsigned request_size = size;
//   if (!write) request_size = READ_PACKET_SIZE;
//   return !::pim_icnt_has_buffer(m_xbar_id, request_size);
// }

// void pim_xbar::xbar_icnt_inject_request_packet(mem_fetch *mf) {
//   unsigned int packet_size = mf->size();
//   if (!mf->get_is_write() && !mf->isatomic()) {
//     packet_size = mf->get_ctrl_size();
//   }
//   // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
//   unsigned destination = mf->get_sub_partition_id();
//   assert(destination < m_pim_config->num_xbars);
//   mf->set_status(IN_ICNT_TO_XBAR,
//                  get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle);
//   if (!mf->get_is_write() && !mf->isatomic())
//     ::pim_icnt_push(m_xbar_id, m_config->mem2device(destination), (void *)mf,
//                     mf->get_ctrl_size());
//   else
//     ::pim_icnt_push(m_xbar_id, m_config->mem2device(destination), (void *)mf,
//                     mf->size());
// }

// bool xbar_memory_interface::full(unsigned size, bool write) const {
//   return m_xbar->xbar_icnt_injection_buffer_full(size, write);
// }

// void xbar_memory_interface::push(mem_fetch *mf) {
//   // m_core->inc_simt_to_mem(mf->get_num_flits(true));
//   m_xbar->xbar_icnt_inject_request_packet(mf);
// }

bool pim_core_cluster::map_layer(pim_layer *layer) {
  for (unsigned i = 0; i < m_config->n_pim_cores_per_cluster; i++) {
    bool issued = false;
    if (layer->name.find("query") == std::string::npos &&
        layer->name.find("key") == std::string::npos &&
        layer->name.find("value") == std::string::npos) {
      return false;
    }

    if (layer->type == pim_layer_type::CONV) {
      issued = m_core[i]->map_layer_conv2d(layer);
    } else if (layer->type == pim_layer_type::MATMUL) {
      issued = m_core[i]->map_layer_linear(layer);
    }
    if (issued) {
      return true;
    }
  }
  return false;
}

void pim_core_cluster::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1D_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

pim_xbar *pim_core_ctx::next_avail_xbar() {
  for (unsigned i = 0; i < m_pim_core_config->num_xbars; i++) {
    if (!m_xbars[i]->mapped) {
      return m_xbars[i];
    }
  }
  assert(0 && "PIM core full\n");
}

bool pim_core_ctx::map_layer_conv2d(pim_layer *layer) {
  assert(layer->type == pim_layer_type::CONV);
  // filter height * input channels
  unsigned rows_total = layer->R * layer->S * layer->C;
  // output channels
  unsigned cols_total = layer->K;

  layer->global_height = rows_total;

  // round up
  unsigned xbar_row_used =
      std::ceil((float)rows_total / m_pim_core_config->xbar_size_y);
  unsigned xbar_col_used =
      std::ceil((float)cols_total / m_pim_core_config->xbar_size_x);

  unsigned xbar_needed = xbar_row_used * xbar_col_used;
  xbar_needed = xbar_needed * m_pim_core_config->num_device_per_weight();

  if (xbar_needed + used_xbars > m_pim_core_config->num_xbars) {
    return false;
  }

  unsigned tot_weight = layer->R * layer->S * layer->C * layer->K *
                        m_pim_core_config->get_data_size_byte();
  unsigned input_size = layer->N * layer->C * layer->H * layer->W *
                        m_pim_core_config->get_data_size_byte();
  unsigned output_size = layer->N * layer->K * layer->P * layer->Q *
                         m_pim_core_config->get_data_size_byte();

  new_addr_type weight_addr = m_gpu->pim_addr;
  m_gpu->pim_addr += tot_weight;
  new_addr_type input_addr = m_gpu->pim_addr;
  m_gpu->pim_addr += input_size;
  new_addr_type output_addr = m_gpu->pim_addr;
  m_gpu->pim_addr += output_size;

  layer->data_size = m_pim_core_config->get_data_size_byte();
  layer->im2col(input_addr);
  layer->input_addr = input_addr;
  layer->output_addr = output_addr;

  std::vector<unsigned> xbars;

  unsigned row_per_xbar = std::ceil((float)rows_total / xbar_row_used);
  assert(row_per_xbar <= m_pim_core_config->xbar_size_y);

  unsigned col_per_xbar = std::ceil((float)cols_total / xbar_col_used);
  assert(col_per_xbar <= m_pim_core_config->xbar_size_x);

  for (unsigned slice_index = 0;
       slice_index < m_pim_core_config->num_device_per_weight();
       slice_index++) {
    unsigned weight_start = weight_addr;

    for (unsigned xbar_col_index = 0; xbar_col_index < xbar_col_used;
         xbar_col_index++) {
      for (unsigned xbar_row_index = 0; xbar_row_index < xbar_row_used;
           xbar_row_index++) {
        pim_xbar *xbar = next_avail_xbar();
        printf("layer %s mapped to core %u xbar %u\n", layer->name.c_str(),
               m_tpc, xbar->m_xbar_id);

        xbar->used_cols = col_per_xbar;
        xbar->used_rows = row_per_xbar;

        xbar->xbar_row_id = xbar_row_index;
        xbar->xbar_col_id = xbar_col_index;

        // point to the first op position
        xbar->op_id = xbar_row_index * row_per_xbar;

        unsigned xbar_weight_size = xbar->used_rows * xbar->used_cols *
                                    m_pim_core_config->get_data_size_byte();
        xbar->weight.addr = weight_start;
        xbar->weight.size = xbar_weight_size;

        weight_start += xbar_weight_size;
        assert(weight_start <= input_addr);  // weight < input < output

        xbar->byte_per_row =
            xbar->used_cols * m_pim_core_config->device_precision / 8;

        // xbar->total_activation = layer->P * layer->Q *
        //                          m_pim_core_config->device_precision /
        //                          m_pim_core_config->dac_precision;

        // suppose device precison 4 bit, dac precision 1 bit -> apply 1 input
        // bit each time
        assert(m_pim_core_config->device_precision %
                   m_pim_core_config->dac_precision ==
               0);

        // xbar->sample_scale_factor =
        //     std::ceil((float)xbar->used_cols / m_pim_core_config->adc_count)
        //     * std::ceil((float)xbar->used_rows /
        // std::pow(2, m_pim_core_config->adc_precision));

        // debugging
        // xbar->total_activation = 8;

        xbar->m_status = XBAR_INITIATED;
        xbar->mapped = true;
        xbar->m_layer = layer;

        used_xbars++;

        layer->m_xbars.push_back(xbar);

        // stats
        unsigned total_devices =
            m_pim_core_config->xbar_size_x * m_pim_core_config->xbar_size_y;
        unsigned used_devices = xbar->used_cols * xbar->used_rows;

        unsigned utilization = 100 * used_devices / total_devices;
        m_pim_stats->xbar_program_efficiency[m_tpc - m_config->n_simt_clusters]
                                            [xbar->m_xbar_id] = utilization;
      }
    }
  }

  // m_running_layers.push_back(layer);
  layer->mapped = true;

  return true;
}

bool pim_core_ctx::map_layer_linear(pim_layer *layer) {
  // filter height * input channels
  unsigned rows_total = layer->R * layer->C;
  // output channels
  unsigned cols_total = layer->S;

  assert(layer->N = layer->K);

  layer->global_height = rows_total;

  // round up
  unsigned xbar_row_used =
      std::ceil((float)rows_total / m_pim_core_config->xbar_size_y);
  unsigned xbar_col_used =
      std::ceil((float)cols_total / m_pim_core_config->xbar_size_x);

  unsigned xbar_needed = xbar_row_used * xbar_col_used;
  xbar_needed = xbar_needed * m_pim_core_config->num_device_per_weight();
  printf("xbar needed %u\n", xbar_needed);

  if (xbar_needed + used_xbars > m_pim_core_config->num_xbars) {
    return false;
  }

  unsigned tot_weight = layer->R * layer->S * layer->C * layer->K *
                        m_pim_core_config->get_data_size_byte();
  unsigned input_size = layer->N * layer->C * layer->H * layer->W *
                        m_pim_core_config->get_data_size_byte();
  unsigned output_size = layer->N * layer->K * layer->P * layer->Q *
                         m_pim_core_config->get_data_size_byte();

  new_addr_type weight_addr = m_gpu->pim_addr;
  m_gpu->pim_addr += tot_weight;
  new_addr_type input_addr = m_gpu->pim_addr;
  m_gpu->pim_addr += input_size;
  new_addr_type output_addr = m_gpu->pim_addr;
  m_gpu->pim_addr += output_size;

  layer->data_size = m_pim_core_config->get_data_size_byte();
  layer->gemm(input_addr);
  layer->input_addr = input_addr;
  layer->output_addr = output_addr;

  std::vector<unsigned> xbars;

  unsigned row_per_xbar = std::ceil((float)rows_total / xbar_row_used);
  assert(row_per_xbar <= m_pim_core_config->xbar_size_y);

  unsigned col_per_xbar = std::ceil((float)cols_total / xbar_col_used);
  assert(col_per_xbar <= m_pim_core_config->xbar_size_x);

  for (unsigned slice_index = 0;
       slice_index < m_pim_core_config->num_device_per_weight();
       slice_index++) {
    unsigned weight_start = weight_addr;

    for (unsigned xbar_col_index = 0; xbar_col_index < xbar_col_used;
         xbar_col_index++) {
      for (unsigned xbar_row_index = 0; xbar_row_index < xbar_row_used;
           xbar_row_index++) {
        pim_xbar *xbar = next_avail_xbar();
        printf("layer %s mapped to core %u xbar %u\n", layer->name.c_str(),
               m_tpc, xbar->m_xbar_id);

        // xbar->output.addr = output_addr;
        // xbar->output.size = output_size;

        // xbar->input.addr = input_addr;
        // xbar->input.size = input_size;

        xbar->used_cols = col_per_xbar;
        xbar->used_rows = row_per_xbar;

        xbar->xbar_row_id = xbar_row_index;
        xbar->xbar_col_id = xbar_col_index;

        // point to the first op position
        xbar->op_id = xbar_row_index * row_per_xbar;

        unsigned xbar_weight_size = xbar->used_rows * xbar->used_cols *
                                    m_pim_core_config->get_data_size_byte();
        xbar->weight.addr = weight_start;
        xbar->weight.size = xbar_weight_size;

        weight_start += xbar_weight_size;
        assert(weight_start <= input_addr);  // weight < input < output

        xbar->byte_per_row =
            xbar->used_cols * m_pim_core_config->device_precision / 8;

        // xbar->total_activation = layer->P * layer->Q *
        //                          m_pim_core_config->device_precision /
        //                          m_pim_core_config->dac_precision;

        // suppose device precison 4 bit, dac precision 1 bit -> apply 1 input
        // bit each time
        assert(m_pim_core_config->device_precision %
                   m_pim_core_config->dac_precision ==
               0);

        // xbar->sample_scale_factor =
        //     std::ceil((float)xbar->used_cols / m_pim_core_config->adc_count)
        //     * std::ceil((float)xbar->used_rows /
        // std::pow(2, m_pim_core_config->adc_precision));

        // debugging
        // xbar->total_activation = 8;

        xbar->m_status = XBAR_INITIATED;
        xbar->mapped = true;
        xbar->m_layer = layer;

        used_xbars++;

        layer->m_xbars.push_back(xbar);

        // stats
        unsigned total_devices =
            m_pim_core_config->xbar_size_x * m_pim_core_config->xbar_size_y;
        unsigned used_devices = xbar->used_cols * xbar->used_rows;

        unsigned utilization = 100 * used_devices / total_devices;
        m_pim_stats->xbar_program_efficiency[m_tpc - m_config->n_simt_clusters]
                                            [xbar->m_xbar_id] = utilization;
      }
    }
  }

  // m_running_layers.push_back(layer);
  layer->mapped = true;
  return true;
}

bool pim_core_ctx::issue_mem(warp_inst_t *inst, active_mask_t active_mask) {
  if (m_ldst_reg[0]->has_free()) {
    warp_inst_t **pipe_reg = m_ldst_reg[0]->get_free();
    assert(pipe_reg);
    **pipe_reg = *inst;
    (*pipe_reg)->issue(active_mask, inst->pim_xbar_id,
                       get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle,
                       0, 0);
    (*pipe_reg)->generate_mem_accesses();
    // (*pipe_reg)->print_m_accessq();

    return true;
  }
  return false;
}

void pim_core_ctx::writeback(warp_inst_t *inst) {
  assert(inst->op == XBAR_SAMPLE_OP);
}

void pim_core_ctx::create_exec_pipeline() {
  unsigned total_pipeline_stages = 1;
  for (unsigned j = 0; j < m_pim_core_config->num_xbars; j++) {
    m_issue_reg.push_back(
        new register_set(total_pipeline_stages, "XBAR_ISSUE"));
    m_result_reg.push_back(
        new register_set(total_pipeline_stages, "XBAR_RESULT"));
  }
  m_ldst_reg.push_back(new register_set(total_pipeline_stages, "LDST"));
  num_result_bus = m_pim_core_config->num_xbars;
  for (unsigned i = 0; i < num_result_bus; i++) {
    this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
  }

  for (unsigned i = 0; i < m_pim_core_config->num_xbars; i++) {
    m_xbars.push_back(new pim_xbar(m_result_reg[i], m_config, this, i));
  }
}

void pim_core_ctx::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1D_sub_stats(css);
}

void pim_core_ctx::get_cache_stats(cache_stats &cs) {
  m_ldst_unit->get_cache_stats(cs);  // Get L1D, L1C, L1T stats
}

void pim_core_stats::print(FILE *fout, unsigned long long tot_cycle) const {
  // for (unsigned tpc = 0; tpc < xbar_program_efficiency.size(); tpc++) {
  //   for (unsigned i = 0; i < xbar_program_efficiency[tpc].size(); i++) {
  //     if (xbar_program_efficiency[tpc][i] == 0) continue;
  //     fprintf(fout, "xbar_program_efficiency[%u][%u]: %u\n", tpc, i,
  //             xbar_program_efficiency[tpc][i]);
  //   }
  // }
  // fprintf(fout, "\n");

  for (unsigned tpc = 0; tpc < xbar_executed_inst.size(); tpc++) {
    for (unsigned i = 0; i < xbar_executed_inst[tpc].size(); i++) {
      if (xbar_executed_inst[tpc][i] == 0) continue;
      fprintf(fout, "xbar_executed_inst[%u][%u]: %u\n", tpc, i,
              xbar_executed_inst[tpc][i]);
    }
  }
  fprintf(fout, "\n");

  for (unsigned tpc = 0; tpc < xbar_active_cycle.size(); tpc++) {
    for (unsigned i = 0; i < xbar_active_cycle[tpc].size(); i++) {
      if (xbar_active_cycle[tpc][i] == 0) continue;
      fprintf(fout, "xbar_active_cycle[%u][%u]: %u [%.2f]\n", tpc, i,
              xbar_active_cycle[tpc][i],
              (float)xbar_active_cycle[tpc][i] / xbar_tot_cycle[tpc][i]);
    }
  }
  fprintf(fout, "\n");

  for (unsigned reason = 0; reason < xbar_stall.size(); reason++) {
    if (xbar_stall[reason] == 0) continue;
    fprintf(fout, "xbar_stall[%u]: %u\n", reason, xbar_stall[reason]);
  }
  fprintf(fout, "\n");
}

void simple_ldst_unit::cycle() {
  writeback();

  for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
    if (m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage + 1]->empty())
      move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);

  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front();
    if (mf->get_type() == WRITE_ACK ||
        (m_config->gpgpu_perfect_mem && mf->get_is_write())) {
      m_core->store_ack(mf);
      m_response_fifo.pop_front();
      delete mf;
    } else {
      assert(!mf->get_is_write());  // L1 cache is write evict, allocate line
                                    // on load miss only

      bool bypassL1D = false;
      if (CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL)) {
        bypassL1D = true;
      } else if (mf->get_access_type() == GLOBAL_ACC_R ||
                 mf->get_access_type() ==
                     GLOBAL_ACC_W) {  // global memory access
        if (m_core->get_config()->gmem_skip_L1D) bypassL1D = true;
      }
      if (bypassL1D) {
        if (m_next_global == NULL) {
          mf->set_status(IN_SHADER_FETCHED,
                         m_core->get_gpu()->gpu_sim_cycle +
                             m_core->get_gpu()->gpu_tot_sim_cycle);
          m_response_fifo.pop_front();
          m_next_global = mf;
        }
      } else {
        if (m_L1D->fill_port_free()) {
          m_L1D->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                              m_core->get_gpu()->gpu_tot_sim_cycle);
          m_response_fifo.pop_front();
        }
      }
    }
  }

  if (m_L1D) {
    m_L1D->cycle();
    if (m_config->m_L1D_config.l1_latency > 0) L1_latency_queue_cycle();
  }

  warp_inst_t &pipe_reg = *m_dispatch_reg;
  enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
  mem_stage_access_type type;
  bool done = true;
  done &= memory_cycle(pipe_reg, rc_fail, type);
  m_mem_rc = rc_fail;

  if (!done) {  // log stall types and return
    assert(rc_fail != NO_RC_FAIL);
    m_stats->gpgpu_n_stall_shd_mem++;
    m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
    return;
  }

  if (!pipe_reg.empty()) {
    unsigned xbar_id = pipe_reg.pim_xbar_id;
    if (pipe_reg.is_load()) {
      bool pending_requests = false;
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
        unsigned reg_id = pipe_reg.out[r];
        if (reg_id > 0) {
          for (auto xbar_id : pipe_reg.m_multicast_xbars) {
            if (m_pending_writes[xbar_id].find(reg_id) !=
                m_pending_writes[xbar_id].end()) {
              if (m_pending_writes[xbar_id][reg_id] > 0) {
                pending_requests = true;
                break;
              } else {
                // this instruction is done already
                m_pending_writes[xbar_id].erase(reg_id);
              }
            }
          }
        }
      }
      if (!pending_requests) {
        // m_core->warp_inst_complete(*m_dispatch_reg);
        for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
          if (m_dispatch_reg->out[r] > 0) {
            for (auto xbar_id : m_dispatch_reg->m_multicast_xbars) {
              m_scoreboard->releaseRegister(xbar_id, m_dispatch_reg->out[r]);
            }
          }
        }
      }
      // m_core->dec_inst_in_pipeline(warp_id);
      m_dispatch_reg->clear();
    } else {
      // stores exit pipeline here
      // m_core->dec_inst_in_pipeline(warp_id);
      // m_core->warp_inst_complete(*m_dispatch_reg);
      m_dispatch_reg->clear();
    }
  }
}

void simple_ldst_unit::issue(register_set &reg_set) {
  warp_inst_t *inst = *(reg_set.get_ready());

  // record how many pending register writes/memory accesses there are for this
  // instruction
  assert(inst->empty() == false);

  inst->op_pipe = MEM__OP;
  if (inst->is_load() and inst->space.get_type() != shared_space) {
    unsigned n_accesses = inst->accessq_count();
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      unsigned reg_id = inst->out[r];
      if (reg_id > 0) {
        for (auto xbar_id : inst->m_multicast_xbars) {
          m_pending_writes[xbar_id][reg_id] += n_accesses;
        }
      }
    }
  }

  // stat collection
  // m_core->mem_instruction_stats(*inst);
  // m_core->incmem_stat(m_core->get_config()->warp_size, 1);
  pipelined_simd_unit::issue(reg_set);
}

void simple_ldst_unit::writeback() {
  // simple writeback
  if (!m_next_wb.empty()) {
    bool insn_completed = false;
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (m_next_wb.out[r] > 0) {
        std::vector<unsigned> multicast_xbars = m_next_wb.m_multicast_xbars;
        if (m_next_wb.space.get_type() != shared_space) {
          for (auto xbar_id : multicast_xbars) {
            assert(m_pending_writes[xbar_id][m_next_wb.out[r]] > 0);
            unsigned still_pending =
                --m_pending_writes[xbar_id][m_next_wb.out[r]];
            if (!still_pending) {
              m_pending_writes[xbar_id].erase(m_next_wb.out[r]);
              m_scoreboard->releaseRegister(xbar_id, m_next_wb.out[r]);

              insn_completed = true;
            }
          }
        } else {  // shared
          for (auto xbar_id : multicast_xbars) {
            m_scoreboard->releaseRegister(xbar_id, m_next_wb.out[r]);
          }
          insn_completed = true;
        }
      }
    }
    if (insn_completed) {
    }

    m_next_wb.clear();
    m_last_inst_gpu_sim_cycle = m_core->get_gpu()->gpu_sim_cycle;
    m_last_inst_gpu_tot_sim_cycle = m_core->get_gpu()->gpu_tot_sim_cycle;
  }
  if (m_L1D && m_L1D->access_ready() && m_next_wb.empty()) {
    mem_fetch *mf = m_L1D->next_access();
    m_next_wb = mf->get_inst();
    delete mf;
  }
}

void simple_ldst_unit::L1_latency_queue_cycle() {
  for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
    if ((l1_latency_queue[j][0]) != NULL) {
      mem_fetch *mf_next = l1_latency_queue[j][0];
      std::list<cache_event> events;
      enum cache_request_status status =
          m_L1D->access(mf_next->get_addr(), mf_next,
                        m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle,
                        events);

      bool write_sent = was_write_sent(events);
      bool read_sent = was_read_sent(events);

      if (status == HIT) {
        assert(!read_sent);
        l1_latency_queue[j][0] = NULL;

        warp_inst_t inst = mf_next->get_inst();
        if (mf_next->get_inst().is_load()) {
          for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
            if (mf_next->get_inst().out[r] > 0) {
              for (auto xbar_id : inst.m_multicast_xbars) {
                assert(m_pending_writes[xbar_id][mf_next->get_inst().out[r]] >
                       0);
                unsigned still_pending =
                    --m_pending_writes[xbar_id][mf_next->get_inst().out[r]];
                if (!still_pending) {
                  m_pending_writes[xbar_id].erase(mf_next->get_inst().out[r]);
                  m_scoreboard->releaseRegister(xbar_id,
                                                mf_next->get_inst().out[r]);
                }
              }
            }
        }

        // For write hit in WB policy
        if (mf_next->get_inst().is_store() && !write_sent) {
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1;

          mf_next->set_reply();

          for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
        }

        if (!write_sent) delete mf_next;

      } else if (status == RESERVATION_FAIL) {
        assert(!read_sent);
        assert(!write_sent);
      } else {
        assert(status == MISS || status == HIT_RESERVED);
        l1_latency_queue[j][0] = NULL;
        if (m_config->m_L1D_config.get_write_policy() != WRITE_THROUGH &&
            mf_next->get_inst().is_store() &&
            (m_config->m_L1D_config.get_write_allocate_policy() ==
                 FETCH_ON_WRITE ||
             m_config->m_L1D_config.get_write_allocate_policy() ==
                 LAZY_FETCH_ON_READ) &&
            !was_writeallocate_sent(events)) {
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1;
          mf_next->set_reply();
          for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
          if (!write_sent && !read_sent) delete mf_next;
        }
      }
    }

    for (unsigned stage = 0; stage < m_config->m_L1D_config.l1_latency - 1;
         ++stage)
      if (l1_latency_queue[j][stage] == NULL) {
        l1_latency_queue[j][stage] = l1_latency_queue[j][stage + 1];
        l1_latency_queue[j][stage + 1] = NULL;
      }
  }
}

void pim_layer::im2col(new_addr_type addr) {
  int new_height = (H + 2 * pad_h - R) / stride_h + 1;
  assert(new_height == (int)P);
  int new_width = (W + 2 * pad_w - S) / stride_w + 1;
  assert(new_width == (int)Q);

  std::vector<new_addr_type> output;
  output.resize(N * new_height * new_width * C * R * S);
  std::vector<new_addr_type> input;
  input.resize(N * C * H * W);
  for (unsigned i = 0; i < N * C * H * W; ++i) {
    // instead of data, save addr
    input[i] = addr + i;
  }

  for (int n = 0; n < (int)N; ++n) {
    for (int h = 0; h < (int)new_height; ++h) {
      for (int w = 0; w < (int)new_width; ++w) {
        for (int c = 0; c < (int)C; ++c) {
          for (int i = 0; i < (int)R; ++i) {
            for (int j = 0; j < (int)S; ++j) {
              int h_pad = h * stride_h + i - pad_h;
              int w_pad = w * stride_w + j - pad_w;
              unsigned out_index = n * new_height * new_width * C * R * S +
                                   h * new_width * C * R * S + w * C * R * S +
                                   c * R * S + i * S + j;  // for loop order

              if (h_pad >= 0 && h_pad < (int)H && w_pad >= 0 &&
                  w_pad < (int)W) {
                output[out_index] =
                    // input[n * C * H * W + c * H * W + h_pad * W + w_pad]; //
                    // NCHW
                    input[n * H * W * C + h_pad * W * C + w_pad * C +
                          c];  // NHWC
              } else {
                output[out_index] = 0;
              }
              // printf("%-8llu ", output[out_index]);
            }
            // printf("\n");
          }
          // printf("\n");
        }
        // printf("\n");
      }
    }
  }
  matmul_addr = output;
  input_size = N * new_height * new_width * C * R * S;
}

void pim_layer::gemm(new_addr_type addr) {
  matmul_addr = std::vector<new_addr_type>(N * C * H * W);
  for (unsigned i = 0; i < N * C * H * W; ++i) {
    matmul_addr[i] = addr + i;
  }

  input_size = N * C * H * W;
}