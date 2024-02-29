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
#include "pim_icnt_wrapper.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "shader_trace.h"
#include "stat-tool.h"
#include "traffic_breakdown.h"
#include "visualizer.h"

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
  m_ldst_unit =

      new simple_ldst_unit(m_icnt, m_mem_fetch_allocator, m_config, m_memory_config,
                           m_stats, m_sid, m_tpc, gpu, dummy_shader, this);
  // m_cache = new simple_ldst_unit(
  //     "L1D", gpu->getShaderCoreConfig()->m_L1D_config, shader_id,
  //     get_shader_normal_cache_id(), m_icnt, m_mem_fetch_allocator,
  //     IN_L1D_MISS_QUEUE, gpu, this);

  // for (unsigned i = 0; i < m_pim_core_config->num_scratchpads; i++) {
  //   char L1D_name[32];
  //   snprintf(L1D_name, 32, "L1D_%03d", i);
  //   m_scratchpads.push_back(new scratchpad(
  //       L1D_name, gpu->getShaderCoreConfig()->m_L1D_config, shader_id,
  //       get_shader_normal_cache_id(), m_icnt, m_mem_fetch_allocator,
  //       IN_SCRATCHPAD, m_pim_core_config->scratchpad_size));
  // }
  m_pim_stats = pim_stats;

  sent_bytes = 0;
  used_xbars = 0;
  core_full = false;

  m_pending_loads = 0;

  unsigned total_pipeline_stages = 1;
  for (unsigned j = 0; j < m_pim_core_config->num_xbars; j++) {
    m_issue_reg.push_back( new register_set(total_pipeline_stages, "XBAR_ISSUE"));
    m_result_reg.push_back(new register_set(total_pipeline_stages, "XBAR_RESULT"));
  }
  m_ldst_reg.push_back(new register_set(total_pipeline_stages, "LDST"));
  num_result_bus = m_pim_core_config->num_xbars;
  for (unsigned i = 0; i < num_result_bus; i++) {
    this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
  }

  for (unsigned i = 0; i < m_pim_core_config->num_xbars; i++) {
    m_xbars.push_back(new pim_xbar(m_result_reg[i], m_config, this, i));
  }

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
  control_cycle();
}

void pim_core_ctx::execute() {
  unsigned cycle = get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  // for (unsigned i = 0; i < num_result_bus; i++) {
  //   *(m_result_bus[i]) >>= 1;
  // }
  for (unsigned n = 0; n < m_pim_core_config->num_xbars; n++) {
    if (!m_xbars[n]->mapped) 
      continue;
    unsigned multiplier = m_xbars[n]->clock_multiplier();
    for (unsigned c = 0; c < multiplier; c++) m_xbars[n]->cycle();
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
  }

  m_ldst_unit->cycle();
  register_set *issue_inst = m_ldst_reg[0];
  warp_inst_t **ready_reg = issue_inst->get_ready();
  if (issue_inst->has_ready() && m_ldst_unit->can_issue(**ready_reg)) {
    m_ldst_unit->issue(*issue_inst);
  }

}

void pim_core_ctx::control_cycle() {
  unsigned cycle = get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  unsigned checked_xbars = 0;

  for (auto xbar : m_xbars) {
    if (!xbar->mapped) 
      continue;
    checked_xbars++;

    if (xbar->m_status == XBAR_INITIATED) {
      unsigned size = 32;
      xbar->m_op_pending_loads.clear();
      xbar->m_op_pending_loads.resize(get_pim_core_config()->xbar_size_y, 0);
      for (unsigned row = 0; row < xbar->used_rows; row++) {
        unsigned byte_per_row = xbar->byte_per_row;
        new_addr_type row_addr = xbar->weight.addr + row * byte_per_row;
        unsigned sent_bytes = 0;
        while (sent_bytes < byte_per_row) {
          new_addr_type addr = row_addr + sent_bytes;
          // to sector
          addr =
              get_gpu()->getShaderCoreConfig()->m_L1D_config.sector_addr(addr);
          if (xbar->addr_to_op.find(addr) == xbar->addr_to_op.end()) {
            xbar->addr_to_op.insert(std::make_pair(addr, std::set<unsigned>()));

            // save to load queue
            xbar->m_load_queue.push(addr);

            xbar->addr_to_op.at(addr).insert(row);
            xbar->m_op_pending_loads[row]++;
          } else {
            if ((xbar->addr_to_op.at(addr)).find(row) ==
                (xbar->addr_to_op.at(addr)).end()) {
              // addr saved, but not for this row
              xbar->addr_to_op.at(addr).insert(row);
              xbar->m_op_pending_loads[row]++;
            }
          }
          sent_bytes += size;
        }
      }
      xbar->m_status = XBAR_PROGRAM;

    } else if (xbar->m_status == XBAR_PROGRAM) {
      while (xbar->m_load_queue.size() > 0) {
        new_addr_type addr = xbar->m_load_queue.front();
        
        bool issued = issue_load(addr, xbar->m_xbar_id);
        if (issued) {
          printf("xbar %u issued load at %u\n", xbar->m_xbar_id, cycle);
          xbar->m_load_queue.pop();
        } else {
          break;
        }
      }
    } else if (xbar->m_status == XBAR_PROGRAMMED) {
      unsigned size = 32;
      unsigned i = 0;
      unsigned op_id = 0;
      unsigned input_repeat = m_pim_core_config->device_precision /
                              m_pim_core_config->dac_precision;
      xbar->m_op_pending_loads.clear();
      while (i < xbar->m_layer->input_size) {
        for (unsigned k = 0; k < input_repeat; k++) {
          assert(op_id == xbar->m_op_pending_loads.size());
          unsigned pending_loads = 0;
          for (unsigned j = 0; j < m_pim_core_config->row_activation_rate;
               j++) {
            new_addr_type addr = xbar->m_layer->input_addr[i + j];
            if ((i + j) == xbar->m_layer->input_size) {
              break;  // out of bound
            }
            if ((i + j) == xbar->used_rows) {
              break;  // out of bound
            }
            if (addr == 0) {
              continue;
            }
            // align to sector
            addr = get_gpu()->getShaderCoreConfig()->m_L1D_config.sector_addr(
                addr);
            if (xbar->addr_to_op.find(addr) == xbar->addr_to_op.end()) {
              xbar->addr_to_op.insert({addr, std::set<unsigned>()});

              // save to load queue
              xbar->m_load_queue.push(addr);

              xbar->addr_to_op.at(addr).insert(op_id);
              pending_loads++;
            } else {
              if ((xbar->addr_to_op.at(addr)).find(op_id) ==
                  (xbar->addr_to_op.at(addr)).end()) {
                // addr saved, but not for this row
                xbar->addr_to_op.at(addr).insert(op_id);
                pending_loads++;
              }
            }
          }
          if (pending_loads == 0) {
            break;  // all 0s. skip
          }
          op_id++;
          xbar->m_op_pending_loads.push_back(pending_loads);
        }
        i += m_pim_core_config->row_activation_rate;
      }
      xbar->total_activation = xbar->m_op_pending_loads.size();
      xbar->m_status = XBAR_COMPUTING;
    } else if (xbar->m_status == XBAR_COMPUTING) {
      if (!xbar->active) 
        continue;
      while (xbar->m_load_queue.size() > 0) {
        new_addr_type addr = xbar->m_load_queue.front();
        
        bool issued = issue_load(addr, xbar->m_xbar_id);
        if (issued) {
          printf("xbar %u issued load at %u\n", xbar->m_xbar_id, cycle);
          xbar->m_load_queue.pop();
        } else {
          break;  // stall
        }
        if (xbar->m_load_queue.size() == 0) {
          printf("xbar %u issued all mf at XBAR_COMPUTING, %u\n",
                 xbar->m_xbar_id, cycle);
        }
      }

    } else if (xbar->m_status == XBAR_DONE) {
      printf("xbar %u done, %u\n", xbar->m_xbar_id, cycle);
      xbar->active = false;
      m_gpu->pim_active = false;
      break;
      pim_layer *next_layer = xbar->m_layer->next_layer;
      if (!next_layer) {
         m_gpu->pim_active = false; // done
      }
      std::vector<unsigned> next_xbars = m_layer_to_xbars.at(next_layer);
      for (auto next_xbar_id : next_xbars) {
        m_xbars[next_xbar_id]->active = true;
      }
      xbar->m_status = XBAR_IDLE;
    }
    break;
  }
}

void pim_core_ctx::issue() {
  unsigned cycle = get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  for (auto xbar : m_xbars) {
    if (!xbar->mapped) 
      continue;
    while (xbar->op_queue.size() > 0) {
      // issue as many as possible
      if (m_issue_reg[xbar->m_xbar_id]->has_free()) {
        warp_inst_t *inst = xbar->op_queue.front();

        warp_inst_t **pipe_reg = m_issue_reg[xbar->m_xbar_id]->get_free();
        assert(pipe_reg);
        **pipe_reg = *inst;
        (*pipe_reg)->issue(
            active_mask_t().set(0), -1,
            get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle, -1, -1);
        xbar->op_queue.pop_front();
        delete inst;
      } else {
        // stall. Unable to issue
        break;
      }
    }
    break;
  }
}

void pim_core_ctx::commit() {
  unsigned cycle = get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  for (unsigned n = 0; n < m_pim_core_config->num_xbars; n++) {
    if (!m_xbars[n]->mapped) 
      continue;
    if (m_result_reg[n]->has_ready()) {
      warp_inst_t **ready_reg = m_result_reg[n]->get_ready();
      (*ready_reg)->clear();

      if ((*ready_reg)->op == XBAR_PROGRAM_OP) {

        printf("xbar %u finished programming %u, %u\n", n,
               m_xbars[n]->programmed_rows, cycle);

        m_xbars[n]->programmed_rows++;
        if (m_xbars[n]->programmed_rows == m_xbars[n]->used_rows) {
          printf("xbar %u done programming, %u\n", n, cycle);
          m_xbars[n]->m_status = XBAR_PROGRAMMED;
          assert(m_xbars[n]->addr_to_op.size() == 0);
        }
      } else if ((*ready_reg)->op == XBAR_COMPUTE_OP) {
        printf("xbar %u done computing, %u, \n",n , cycle);
      } else if ((*ready_reg)->op == XBAR_SAMPLE_OP) {
        printf("xbar %u done sampling, %u, %u\n", n,
               m_xbars[n]->done_activation, cycle);
        m_xbars[n]->done_activation++;
        if (m_xbars[n]->total_activation == m_xbars[n]->done_activation) {
          printf("xbar %u done sampling, %u\n", n, cycle);
          m_xbars[n]->m_status = XBAR_DONE;
        }
      }
    }
  }
}

bool pim_core_ctx::response_buffer_full() const {
  return m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size;
}

void pim_core_ctx::accept_response(mem_fetch *mf) { 
  m_ldst_unit->fill(mf); 
}

void pim_core_ctx::record_load_done(warp_inst_t *inst) {
  unsigned xbar_id = inst->pim_xbar_id;

  new_addr_type addr = inst->get_addr(0);
  assert(m_xbars[xbar_id]->addr_to_op.find(addr) !=
         m_xbars[xbar_id]->addr_to_op.end());
  std::set<unsigned> ops = m_xbars[xbar_id]->addr_to_op[addr];
  for (auto op: ops) {
    assert(m_xbars[xbar_id]->m_op_pending_loads[op] != 0);
    m_xbars[xbar_id]->m_op_pending_loads[op]--;

    if (m_xbars[xbar_id]->m_op_pending_loads[op] == 0) {
      warp_inst_t *new_inst = new warp_inst_t(m_config);
      switch(m_xbars[xbar_id]->m_status) {
        case XBAR_PROGRAM:
          new_inst->op = XBAR_PROGRAM_OP;
          new_inst->latency = m_pim_core_config->program_latency;
          break;
        case XBAR_COMPUTING:
          new_inst->op = XBAR_COMPUTE_OP;
          new_inst->latency = m_pim_core_config->integrate_latency;
          break;
        default:
          assert(0 && "Invalid xbar status");
      }
      
      m_xbars[xbar_id]->op_queue.push_back(new_inst);
      // add sample op after compute op
      if (m_xbars[xbar_id]->m_status == XBAR_COMPUTING) {
        warp_inst_t *new_inst = new warp_inst_t(m_config);
        new_inst->op = XBAR_SAMPLE_OP;
        new_inst->latency = m_pim_core_config->sample_latency;
        m_xbars[xbar_id]->op_queue.push_back(new_inst);
      }
    }
  }
  m_xbars[xbar_id]->addr_to_op.erase(addr);
  assert(m_pending_loads != 0);
  m_pending_loads--;
  
}

void pim_core_cluster::core_cycle() {
  for (unsigned i = 0; i < m_config->n_pim_cores_per_cluster; i++)
    m_core[i]->cycle();
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
  if ((*ready_reg)->op == XBAR_COMPUTE_OP) {
    computing = true;
  } else if ((*ready_reg)->op == XBAR_SAMPLE_OP) {
    sampling = true;
  } else if ((*ready_reg)->op == XBAR_PROGRAM_OP) {
    programming = true;
  }
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

void pim_core_cluster::map_layer(pim_layer *layer) {
  for (unsigned i = 0; i < m_config->n_pim_cores_per_cluster; i++) {
    if (m_core[i]->can_issue_layer(layer)) { 
      if (layer->type == CONV2D) {
        m_core[i]->map_layer_conv2d(layer);
      }
    }
  }
}

pim_xbar *pim_core_ctx::next_avail_xbar() {
  for (unsigned i = 0; i < m_pim_core_config->num_xbars; i++) {
    if (!m_xbars[i]->mapped) {
      return m_xbars[i];
    }
  }
  assert(0 && "PIM core full\n");
}

void pim_core_ctx::map_layer_conv2d(pim_layer *layer) {
  assert(layer->type == CONV2D);
  // filter height * input channels
  unsigned rows_total = layer->R * layer->S * layer->C;
  // output channels
  unsigned cols_total = layer->K;

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

  // round up
  unsigned xbar_row_used =
      std::ceil((float)rows_total / m_pim_core_config->xbar_size_y);
  unsigned xbar_col_used =
      std::ceil((float)cols_total / m_pim_core_config->xbar_size_x);
  unsigned xbar_needed = xbar_row_used * xbar_col_used;
  std::vector<unsigned> xbars;

  
  for (unsigned j = 0; j < m_pim_core_config->num_device_per_weight(); j++) {
    unsigned mapped_rows = 0;
    unsigned mapped_cols = 0;

    unsigned weight_start = weight_addr;
    for (unsigned i = 0; i < xbar_needed; i++) {
      pim_xbar *xbar = next_avail_xbar();

      xbar->output.addr = output_addr;
      xbar->output.size = output_size;

      xbar->input.addr = input_addr;
      xbar->input.size = input_size;

      if (cols_total - mapped_cols >= m_pim_core_config->xbar_size_x) {
        xbar->used_cols += m_pim_core_config->xbar_size_x;
        mapped_cols += m_pim_core_config->xbar_size_x;
      } else {
        xbar->used_cols += cols_total - mapped_cols;
        mapped_cols = cols_total;
      }

      if (rows_total - mapped_rows >= m_pim_core_config->xbar_size_y) {
        xbar->used_rows += m_pim_core_config->xbar_size_y;
      } else {
        xbar->used_rows += rows_total - mapped_rows;
      }

      unsigned xbar_weight_size = xbar->used_rows * xbar->used_cols *
                             m_pim_core_config->get_data_size_byte();
      xbar->weight.addr = weight_start;
      xbar->weight.size = xbar_weight_size;

      weight_start += xbar_weight_size;
      assert(weight_start <= input_addr); // weight < input < output

      // reset cols counter if all cols are assigned and there are more rows
      if (mapped_cols == cols_total) {
        mapped_rows += xbar->used_rows;
        mapped_cols = 0;
      }

      xbar->byte_per_row =
          xbar->used_cols * m_pim_core_config->device_precision / 8;

      // xbar->total_activation = layer->P * layer->Q *
      //                          m_pim_core_config->device_precision /
      //                          m_pim_core_config->dac_precision;

      // suppose device precison 4 bit, dac precision 1 bit -> apply 1 input bit
      // each time
      assert(m_pim_core_config->device_precision %
                 m_pim_core_config->dac_precision ==
             0);

      // xbar->sample_scale_factor =
      //     std::ceil((float)xbar->used_cols / m_pim_core_config->adc_count) *
      //     std::ceil((float)xbar->used_rows /
      // std::pow(2, m_pim_core_config->adc_precision));

      // debugging
      // xbar->total_activation = 8;

      xbar->m_status = XBAR_INITIATED;
      xbar->mapped = true;
      xbar->m_layer = layer;
      if (layer->m_layer_id == 0) {
        xbar->active = true;
      }

      used_xbars++;
      if (used_xbars == m_pim_core_config->num_xbars) {
        core_full = true;
      }
      m_running_layers.push_back(layer);
      xbars.push_back(xbar->m_xbar_id);

      // stats
      unsigned total_devices =
          m_pim_core_config->xbar_size_x * m_pim_core_config->xbar_size_y;
      unsigned used_devices = xbar->used_cols * xbar->used_rows;

      unsigned utilization = 100 * used_devices / total_devices;
      m_pim_stats->xbar_program_efficiency[xbar->m_xbar_id] = utilization;
    }
    assert(mapped_rows == rows_total);
  }

  m_layer_to_xbars.insert(std::make_pair(layer, xbars));
}

bool pim_core_ctx::can_issue_layer(pim_layer *layer) {
  // filter height * input channels
  unsigned rows_total = layer->R * layer->S * layer->C;
  // output channels
  unsigned cols_total = layer->K;

  unsigned xbar_row_used =
      std::ceil((float)rows_total / m_pim_core_config->xbar_size_y);
  unsigned xbar_col_used =
      std::ceil((float)cols_total / m_pim_core_config->xbar_size_x);
  unsigned xbar_needed = xbar_row_used * xbar_col_used;
  xbar_needed = xbar_needed * m_pim_core_config->num_device_per_weight();

  if (xbar_needed + used_xbars > m_pim_core_config->num_xbars) {
    return false;
  } else {
    return true;
  }
}

bool pim_core_ctx::issue_load(new_addr_type addr, unsigned xbar_id) {
  if (m_ldst_reg[0]->has_free()) {
    warp_inst_t *inst = new warp_inst_t(m_config);
    inst->op = LOAD_OP;
    inst->cache_op = CACHE_ALL;
    inst->data_size = 4;
    inst->set_addr(0, addr);
    inst->space.set_type(global_space);
    inst->pim_xbar_id = xbar_id;
    warp_inst_t **pipe_reg = m_ldst_reg[0]->get_free();
    assert(pipe_reg);
    **pipe_reg = *inst;
    (*pipe_reg)->issue(active_mask_t().set(0), 0,
                       get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle,
                       0, 0);
    delete inst;
    inst = NULL;
    (*pipe_reg)->generate_mem_accesses();
    assert((*pipe_reg)->accessq_count() == 1);
    m_pending_loads++;

    return true;
  }
  return false;
}

void pim_core_stats::print(FILE *fout, unsigned long long tot_cycle) const {
  fprintf(fout, "xbar_program_cycle: \n");
  for (unsigned i = 0; i < xbar_program_cycle.size(); i++) {
    if (xbar_program_cycle[i] == 0) continue;
    fprintf(fout, "xbar_tot_program_cycle[%u]: %u\n", i, xbar_program_cycle[i]);
  }
  fprintf(fout, "\n");

  for (unsigned i = 0; i < xbar_integrate_cycle.size(); i++) {
    if (xbar_integrate_cycle[i] == 0) continue;
    fprintf(fout, "xbar_tot_integrate_cycle[%u]: %u\n", i,
            xbar_integrate_cycle[i]);
  }
  fprintf(fout, "\n");

  for (unsigned i = 0; i < xbar_sample_cycle.size(); i++) {
    if (xbar_sample_cycle[i] == 0) continue;
    fprintf(fout, "xbar_tot_sample_cycle[%u]: %u\n", i, xbar_sample_cycle[i]);
  }
  fprintf(fout, "\n");

  for (unsigned i = 0; i < xbar_program_efficiency.size(); i++) {
    if (xbar_program_efficiency[i] == 0) continue;
    fprintf(fout, "xbar_program_efficiency[%u]: %u\n", i,
            xbar_program_efficiency[i]);
  }
  fprintf(fout, "\n");

  for (unsigned i = 0; i < xbar_active_cycle.size(); i++) {
    if (xbar_active_cycle[i] == 0) continue;
    fprintf(fout, "xbar_active_cycle[%u]: %u [%.2f]\n", i, xbar_active_cycle[i],
            (float)xbar_active_cycle[i] / tot_cycle);
  }
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
    unsigned warp_id = pipe_reg.warp_id();
    if (pipe_reg.is_load()) {
      bool pending_requests = false;
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
        unsigned reg_id = pipe_reg.out[r];
        if (reg_id > 0) {
          if (m_pending_writes[warp_id].find(reg_id) !=
              m_pending_writes[warp_id].end()) {
            if (m_pending_writes[warp_id][reg_id] > 0) {
              pending_requests = true;
              break;
            } else {
              // this instruction is done already
              m_pending_writes[warp_id].erase(reg_id);
            }
          }
        }
      }
      if (!pending_requests) {
        // m_core->warp_inst_complete(*m_dispatch_reg);
        // m_scoreboard->releaseRegisters(m_dispatch_reg);
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
  // stat collection
  // m_core->mem_instruction_stats(*inst);
  // m_core->incmem_stat(m_core->get_config()->warp_size, 1);
  pipelined_simd_unit::issue(reg_set);
}

void simple_ldst_unit::writeback() {
  // simple writeback
  if (!m_next_wb.empty()) {
    if (m_pim_core->m_xbars[m_next_wb.pim_xbar_id]->op_queue.size() < 10) {
      printf("writeback 0x%llx\n", m_next_wb.get_addr(0));
      fflush(stdout);
      m_pim_core->record_load_done(&m_next_wb);
      m_next_wb.clear();
      m_last_inst_gpu_sim_cycle = m_core->get_gpu()->gpu_sim_cycle;
      m_last_inst_gpu_tot_sim_cycle = m_core->get_gpu()->gpu_tot_sim_cycle;
    }
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
        if (m_pim_core->m_xbars[mf_next->get_inst().pim_xbar_id]->op_queue.size() < 10) {
        assert(!read_sent);
        l1_latency_queue[j][0] = NULL;

        warp_inst_t inst = mf_next->get_inst();
        m_pim_core->record_load_done(&inst);
        // if (mf_next->get_inst().is_load()) {
        //   for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
        //     if (mf_next->get_inst().out[r] > 0) {
        //       assert(m_pending_writes[mf_next->get_inst().warp_id()]
        //                              [mf_next->get_inst().out[r]] > 0);
        //       unsigned still_pending =
        //           --m_pending_writes[mf_next->get_inst().warp_id()]
        //                             [mf_next->get_inst().out[r]];
        //       if (!still_pending) {
        //         m_pending_writes[mf_next->get_inst().warp_id()].erase(
        //             mf_next->get_inst().out[r]);
        //         m_scoreboard->releaseRegister(mf_next->get_inst().warp_id(),
        //                                       mf_next->get_inst().out[r]);
        //         m_core->warp_inst_complete(mf_next->get_inst());
        //       }
        //     }
        // }

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
        }

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