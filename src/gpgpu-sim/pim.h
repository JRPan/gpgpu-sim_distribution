// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda, Vijay Kandiah, Nikos Hardavellas, 
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#ifndef PIM_H
#define PIM_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "../abstract_hardware_model.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_fetch.h"
#include "scoreboard.h"
#include "stack.h"
#include "stats.h"
#include "shader.h"

class pim_tile;

enum pim_data_type {
  INT8_TYPE,
  INT16_TYPE,
  INT32_TYPE,
  INT64_TYPE,
  FP16_TYPE,
  FP32_TYPE,
  FP64_TYPE,
  FP128_TYPE,
  DP64_TYPE,
  DP128_TYPE,
  DP256_TYPE,
  NUM_DATA_TYPES
};

enum tile_status {
  TILE_INITIATED,
  TILE_PROGRAM,
  TILE_LOAD_ROW_ISSUED,
  TILE_ROW_PROGRAMMING,
  TILE_PROGRAMMED,
  TILE_LOAD_COL_ISSUED,
  TILE_COMPUTING,
  TILE_STALL_SAMPLE,
  TILE_SAMPLE,
  TILE_DONE,
  TILE_NUM_STATUS
};
class pim_core_config;

unsigned data_type_to_size(pim_data_type type);

class pim_layer {
 public:
  pim_layer() {};
  unsigned N;
  unsigned C;
  unsigned H;
  unsigned W;
  unsigned K;
  unsigned P;
  unsigned Q;
  unsigned R;
  unsigned S;
  unsigned pad_h;
  unsigned pad_w;
  unsigned stride_h;
  unsigned stride_w;
  unsigned dilation_h;
  unsigned dilation_w;
  unsigned group;
};

class pim_core_ctx : public core_t {
 public:
  // creator:
  pim_core_ctx(class gpgpu_sim *gpu, class pim_core_cluster *cluster,
                  unsigned shader_id, unsigned tpc_id,
                  const shader_core_config *config,
                  const memory_config *mem_config, shader_core_stats *stats);

  // used by pim_core_cluster:
  // modifiers
  void cycle();
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed);
  // void issue_block2core(class kernel_info_t &kernel);

  // void cache_flush();
  // void cache_invalidate();
  // void accept_fetch_response(mem_fetch *mf);
  void accept_response(mem_fetch *mf);
  // void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                  //  warp_set_t warps);
  // void set_kernel(kernel_info_t *k) {
  //   assert(k);
  //   m_kernel = k;
  // }
  void set_layer(pim_layer *layer) {
    assert(layer);
    m_layer = layer;
  }

  void memory_cycle();
  void issue();
  void execute();
  void commit();
  void handle_response_fifo();
  mem_fetch *generate_mf(new_addr_type addr);
  void process_program_buffer(unsigned tile_id, mem_fetch *mf);
  void process_input_buffer(unsigned tile_id, mem_fetch *mf);
  int test_res_bus(int latency);

  static const unsigned MAX_ALU_LATENCY = 512;
  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;
 protected:
  pim_core_config *m_pim_core_config;
  pim_layer * m_layer;

 public:
  void inc_simt_to_mem(unsigned n_flits) {
    m_stats->n_simt_to_mem[m_sid] += n_flits;
  }
  PowerscalingCoefficients *scaling_coeffs;
  bool response_buffer_full() const;
  pim_layer *get_layer() { return m_layer; }
  unsigned get_sid() const { return m_sid; }

  virtual void warp_exit(unsigned warp_id);

  virtual bool warp_waiting_at_barrier(unsigned warp_id) const;
  void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;
                              
  virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);


  // pure virtual methods implemented based on the current execution mode
  // (execution-driven vs trace-driven)
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  virtual void func_exec_inst(warp_inst_t &inst) = 0;

  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu) = 0;

  virtual void create_shd_warp() = 0;

  virtual const warp_inst_t *get_next_inst(unsigned warp_id,
                                           address_type pc) = 0;
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc) = 0;
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI) = 0;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // general information
  unsigned m_sid;  // shader id
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  class pim_core_cluster *m_cluster;
  std::vector<register_set> m_issue_reg;
  std::vector<register_set> m_result_reg;

  // statistics
  shader_core_stats *m_stats;

  // interconnect interface
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

  unsigned mf_size = 32;
 private:
  unsigned sent_bytes;
  unsigned layer_mapped;
  
  l1_cache *m_L1D;
  std::list<mem_fetch *> m_response_fifo;
  std::vector<unsigned> m_pending_loads;
  std::unordered_map<mem_fetch *, unsigned> m_loads;
  std::vector<pim_tile *> m_tiles;
  new_addr_type weight_addr = 0xffff7fffffffffff;
  new_addr_type input_addr =  0xffff8fffffffffff;
};

class pim_core_cluster : public simt_core_cluster {
 public:
  pim_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                    const shader_core_config *config,
                    const memory_config *mem_config, shader_core_stats *stats,
                    memory_stats_t *mstats) : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {};

  void core_cycle();
  void icnt_cycle();

  // void reinit();
  // void cache_flush();
  // void cache_invalidate();
  // bool icnt_injection_buffer_full(unsigned size, bool write);
  // void icnt_inject_request_packet(class mem_fetch *mf);

  // // for perfect memory interface
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  }
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf);
  }
  gpgpu_sim *get_gpu() { return m_gpu; }
  // virtual void create_pim_core_ctx() = 0;

//  protected:
//   unsigned m_cluster_id;
//   gpgpu_sim *m_gpu;
//   const shader_core_config *m_config;
//   shader_core_stats *m_stats;
//   memory_stats_t *m_memory_stats;
  pim_core_ctx **m_core;
//   const memory_config *m_mem_config;

//   unsigned m_cta_issue_next_core;
//   std::list<unsigned> m_core_sim_order;
  std::list<mem_fetch *> m_response_fifo;
};

class pim_core_config : public core_config {
 public:
  pim_core_config(gpgpu_context *ctx) : core_config(ctx) {
    num_tiles = 1;
    tile_size_x = 256;
    tile_size_y = 256;
    adc_count = 4;
    adc_precision;
    dac_presicion;
    row_activation_rate;


    program_latency = 10;
    integrate_latency = 10;
    sample_latency = 500;

    device_precision = 4;
    gpgpu_ctx = ctx; 
  }

  void init() {}
  void reg_options(class OptionParser *opp);

  gpgpu_context *gpgpu_ctx;

  unsigned num_tiles;
  unsigned tile_size_x;
  unsigned tile_size_y;
  unsigned dac_count;

  unsigned program_latency;
  unsigned integrate_latency;
  unsigned sample_latency;

  pim_data_type data_type;

  unsigned device_precision;
  unsigned adc_precision;
  unsigned dac_presicion;
  unsigned row_activation_rate;

  unsigned byte_per_row();
  unsigned get_data_size();
};

class pim_memory_interface : public mem_fetch_interface {
 public:
  pim_memory_interface(pim_core_ctx *core, pim_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write);
  }
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->icnt_inject_request_packet(mf);
  }

 private:
  pim_core_ctx *m_core;
  pim_core_cluster *m_cluster;
};

class pim_tile : public pipelined_simd_unit {
 public:
  pim_tile(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id)
      : pipelined_simd_unit(result_port, config, 1024, core,
                            issue_reg_id) {
    m_name = "TILE";
    m_status = TILE_INITIATED;
    m_tile_id = issue_reg_id;
    used_rows = 0;
    used_cols = 0;
    byte_per_row = 0;
    sent_bytes = 0;
    programmed_rows = 0;
    done_activation = 0;
    total_activation = 0;

    sampling = false;
    computing = false;
  }

  virtual bool can_issue(const warp_inst_t &inst) const {
    if (inst.op == TILE_SAMPLE_OP) {
      return !sampling;
    } else if (inst.op == TILE_COMPUTE_OP) {
      return !computing;
    } 

    return pipelined_simd_unit::can_issue(inst);
  }

  virtual void cycle() {
    if (!m_pipeline_reg[0]->empty()) {
      if (m_pipeline_reg[0]->op == TILE_SAMPLE_OP) {
        sampling = false;
      } else if (m_pipeline_reg[0]->op == TILE_COMPUTE_OP) {
        computing = false;
      }
    }
    pipelined_simd_unit::cycle();
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return false; }
  tile_status m_status;
  unsigned used_rows, used_cols;
  unsigned m_tile_id;
  unsigned byte_per_row;
  unsigned sent_bytes;
  unsigned programmed_rows;
  unsigned done_activation;
  unsigned total_activation;

  bool sampling;
  bool computing;
};


#endif