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
#include "shader.h"
#include "shader_trace.h"
#include "stat-tool.h"
#include "traffic_breakdown.h"
#include "visualizer.h"

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

pim_core_ctx::pim_core_ctx(class gpgpu_sim *gpu,
                           class pim_core_cluster *cluster, unsigned shader_id,
                           unsigned tpc_id, const shader_core_config *config,
                           const memory_config *mem_config,
                           shader_core_stats *stats)
    : core_t(gpu, NULL, config->warp_size, config->n_thread_per_shader) {
    m_sid = shader_id;
    m_tpc = tpc_id;
    m_gpu = gpu;
    m_stats = stats;
    m_cluster = cluster;
    m_config = config;
    m_memory_config = mem_config;
    m_icnt = new pim_memory_interface(this, m_cluster);
    m_pim_core_config = new pim_core_config(NULL);
    m_layer = new pim_layer();
    m_L1D = new l1_cache("L1D", gpu->getShaderCoreConfig()->m_L1D_config, shader_id,
                         get_shader_normal_cache_id(), m_icnt,
                         m_mem_fetch_allocator, IN_L1D_MISS_QUEUE, gpu);
    m_mem_fetch_allocator = new shader_core_mem_fetch_allocator(m_sid, m_tpc, m_memory_config);
    m_layer->N = 1;
    m_layer->C = 3;
    m_layer->H = 224;
    m_layer->W = 224;
    m_layer->K = 64;
    m_layer->P = 112;
    m_layer->Q = 112;
    m_layer->R = 7;
    m_layer->S = 7;
    m_layer->pad_h = 3;
    m_layer->pad_w = 3;
    m_layer->stride_h = 2;
    m_layer->stride_w = 2;
    m_layer->dilation_h = 1;
    m_layer->dilation_w = 1;
    m_layer->group = 1;

    sent_bytes = 0;
    layer_mapped = false;

    for (unsigned i = 0; i < m_pim_core_config->num_tiles; i++) {
      m_tiles.push_back(new pim_tile(NULL, i));
    }

    assert(m_pim_core_config->byte_per_row() % mf_size == 0);
    m_pending_loads.resize(m_pim_core_config->tile_size_y,
                           //  m_pim_core_config->byte_per_row() / mf_size);
                           1);

    m_tiles[0]->used_rows = m_layer->S * m_layer->C;
    m_tiles[0]->used_columes =
        m_layer->R * m_layer->K * m_pim_core_config->bit_precision / 8;
    m_tiles[0]->byte_per_row = m_layer->R * m_layer->K;
}

void pim_core_ctx::warp_exit(unsigned warp_id) {

}

bool pim_core_ctx::warp_waiting_at_barrier(unsigned warp_id) const {
  return false;
}

// cluster

void pim_core_ctx::issue_warp(register_set &pipe_reg_set,
                                 const warp_inst_t *next_inst,
                                 const active_mask_t &active_mask,
                                 unsigned warp_id, unsigned sch_id) {
                                 }
void pim_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                 unsigned end_thread, unsigned ctaid,
                                 int cta_size, kernel_info_t &kernel) {
                                 }

void pim_core_ctx::cycle() {
  read_out();
  compute();
  program();
  memory_cycle();
}

void pim_core_ctx::program() {
  for (unsigned i = 0; i < m_pim_core_config->num_tiles; i++) {
    pim_tile *tile = m_tiles[i];

    if (tile->program_in_progress()) {
      if (tile->program_latency_cycle != 0) {
        tile->program_latency_cycle--;
        continue;
      } else {
        tile->finish_programming_row();

        if (tile->check_all_programmed()) {
          tile->set_tile_programmed();
        }
      }
    }

    // start programming
    if (tile->program_queue.size() != 0) {
      tile->program_row();
    }
  }
}

void pim_core_ctx::memory_cycle() {
  m_L1D->cycle();
  // load filters from memory hierarchy

  for (auto tile : m_tiles) {
    // if(tile->check_all_programmed()) {
    //   continue;
    // }

    if (tile->m_status == TILE_PROGRAMMED) {
      // load input
      new_addr_type addr = input_addr + 0;
      mem_fetch *mf = generate_mf(addr);
      std::list<cache_event> events;
      enum cache_request_status status = m_L1D->access(
          mf->get_addr(), mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
          events);
      if (status == RESERVATION_FAIL) {
        continue;
      }
      if (status != HIT) {
        m_loads.insert(std::make_pair(mf, tile->m_tid));
        tile->m_input_buffer.insert(mf);
      }
    } else if (tile->m_status == TILE_ROW_PROGRAMMED ||
               tile->m_status == TILE_INITIATED) {
      if (tile->row_all_loaded()) {
        continue;
      }
      new_addr_type addr = m_config->m_L1D_config.block_addr(
          weight_addr + tile->byte_per_row * tile->programmed_rows +
          tile->sent_bytes);
      mem_fetch *mf = generate_mf(addr);
      std::list<cache_event> events;
      enum cache_request_status status = m_L1D->access(
          mf->get_addr(), mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
          events);
      if (status == RESERVATION_FAIL) {
        continue;
      }
      if (status != HIT) {
        tile->m_program_buffer.insert(mf);
        m_loads.insert(std::make_pair(mf, tile->m_tid));
      }
      tile->sent_bytes += mf_size;
    }
  }

  // unsigned total_bytes = m_layer->C * m_layer->R * m_layer->S * m_layer->K *
  //                        m_pim_core_config->get_data_size();

  // while (!m_icnt->full(mf_size, false) && sent_bytes < total_bytes) {
  //   new_addr_type addr = weight_addr + sent_bytes;
  //   mem_fetch *mf = generate_mf(addr);
  //   // m_icnt->push(mf);

  //   std::list<cache_event> events;
  //   enum cache_request_status status =
  //       m_L1D->access(mf->get_addr(), mf,
  //                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);

  //   unsigned byte_per_row = m_tiles[0]->used_columes * m_pim_core_config->bit_precision / 8;
  //   unsigned row = sent_bytes / byte_per_row;
  //   if (m_loads.find(mf) == m_loads.end()) {
  //     m_loads.insert(std::make_pair(mf, std::set<unsigned>()));
  //   } 
  //   m_loads.at(mf).insert(row);

  //   if (status == RESERVATION_FAIL) {
  //     break;
  //   }
  //   assert(status == MISS || status == SECTOR_MISS);
  //   sent_bytes += mf_size;
  // }

  handle_response_fifo();
}

void pim_core_ctx::compute() {
  for (auto tile : m_tiles) {
    if (tile->m_status == TILE_COMPUTING) {
      if (tile->compute_latency_cycle != 0) {
        tile->compute_latency_cycle--;
        continue;
      } else {

      }
    }

  }
}

void pim_core_ctx::read_out() {

}

void pim_core_ctx::handle_response_fifo() {
  // cycle response fifo from L2
  // fill mf into L1D
  // send to program queue if ready
  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front();
    if (m_L1D->fill_port_free()) {
      m_L1D->fill(mf, m_gpu->gpu_sim_cycle +
                          m_gpu->gpu_tot_sim_cycle);
      unsigned tile_id = m_loads.at(mf);
      m_loads.erase(mf);
      if (mf->get_addr() >= m_config->m_L1D_config.block_addr(weight_addr) &&
          mf->get_addr() < m_config->m_L1D_config.block_addr(input_addr)) {
        process_program_buffer(tile_id, mf);
      } else if (mf->get_addr() >= m_config->m_L1D_config.block_addr(input_addr)) {
        process_input_buffer(tile_id, mf);
      }

      printf("received mf at addr %llx, data size = %u\n", mf->get_addr(), mf->get_data_size());
      m_response_fifo.pop_front();
    }
  }
}

mem_fetch *pim_core_ctx::generate_mf(new_addr_type addr) {
  unsigned chunk = (addr & 127) / 32;
  mem_access_sector_mask_t sector_mask;
  sector_mask.set(chunk);

  new_addr_type block_address =
      m_gpu->getShaderCoreConfig()->m_L1D_config.block_addr(addr);
    
  unsigned index = addr - block_address;
  mem_access_byte_mask_t byte_mask;
  byte_mask.reset();
  for (unsigned i = 0; i < mf_size; i++) byte_mask.set(index + i);

  active_mask_t active_mask;
  active_mask.set();

  mem_fetch *mf = m_mem_fetch_allocator->alloc(
      block_address, GLOBAL_ACC_R, active_mask, byte_mask, sector_mask, mf_size,
      false, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, m_sid, -1,
      NULL);

  return mf;
}

void pim_core_ctx::process_program_buffer(unsigned tile_id, mem_fetch *mf) {
  for (auto ld = m_tiles[tile_id]->m_program_buffer.begin();
       ld != m_tiles[tile_id]->m_program_buffer.end();) {
    if ((*ld)->get_addr() == mf->get_addr()) {
      ld = m_tiles[tile_id]->m_program_buffer.erase(ld);
    } else {
      ld++;
    }
  }

  if (m_tiles[tile_id]->m_program_buffer.empty()) {
          m_tiles[tile_id]->program_queue.push_back(0);
  }
}

void pim_core_ctx::process_input_buffer(unsigned tile_id, mem_fetch *mf) {
  for (auto ld = m_tiles[tile_id]->m_input_buffer.begin();
       ld != m_tiles[tile_id]->m_input_buffer.end();) {
    if ((*ld)->get_addr() == mf->get_addr()) {
      ld = m_tiles[tile_id]->m_input_buffer.erase(ld);
    } else {
      ld++;
    }
  }

  if (m_tiles[tile_id]->m_input_buffer.empty()) {
          // m_tiles[tile_id]->program_queue.push_back(0);
  }
}

bool pim_core_ctx::response_buffer_full() const {
  return m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size;
}

void pim_core_ctx::accept_response(mem_fetch *mf) {
  mf->set_status(
      IN_SHADER_LDST_RESPONSE_FIFO,
      m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  m_response_fifo.push_back(mf);
}

void pim_core_cluster::core_cycle() {
  for(unsigned i = 0; i < m_config->n_pim_cores_per_cluster; i++)
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
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    // m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
    m_response_fifo.push_back(mf);
    m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
  }
}

unsigned pim_core_config::get_data_size() {
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

unsigned pim_core_config::byte_per_row() {
  return get_data_size() * tile_size_x * bit_precision / 8;
}