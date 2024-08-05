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
#include <unordered_map>
#include <queue>

#include "../abstract_hardware_model.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_fetch.h"
#include "scoreboard.h"
#include "stack.h"
#include "stats.h"
#include "shader.h"
#include "pim_icnt_wrapper.h"

class pim_xbar;
class simple_ldst_unit;
class pim_core_config;
class pim_core_stats;

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

enum pim_layer_type {
  CONV = 1,
  CONV2D, // to be removed
  LINEAR,
  INPUT,
  RELU,
  MAXPOOL,
  ADD,
  GLOBAL_AVG_POOL,
  OUTPUT,
  UNDEFINED,
  NUM_LAYER_TYPES
};

enum xbar_status {
  XBAR_INITIATED,
  XBAR_PROGRAM,
  // XBAR_LOAD_ROW_ISSUED,
  // XBAR_ROW_PROGRAMMING,
  XBAR_PROGRAMMED,
  // XBAR_LOAD_COL_ISSUED,
  XBAR_COMPUTING,
  // XBAR_STALL_SAMPLE,
  // XBAR_SAMPLE,
  XBAR_DONE,
  XBAR_IDLE,
  XBAR_NUM_STATUS
};

class buffer {
 public:
 buffer() {
    addr = 0;
    size = 0;
  }
  buffer(new_addr_type _addr, unsigned _size) {
    addr = _addr;
    size = _size;
  }

  new_addr_type addr;
  unsigned size;
};

class pim_layer {
 public:
  pim_layer() {
    N = 0;
    C = 0;
    H = 0;
    W = 0;
    K = 0;
    P = 0;
    Q = 0;
    R = 0;
    S = 0;
    pad_h = 0;
    pad_w = 0;
    stride_h = 0;
    stride_w = 0;
    dilation_h = 0;
    dilation_w = 0;
    group = 0;
    data_size = 0;
    input_size = 0;
    output_addr = 0;
    mapped = false;
    pending_next = 0;
    active = false;
  };
  ~pim_layer() {}
  // void im2col(float *input, float *&output, int N, int C, int H, int W,
  //             int R, int S, int stride_h, int stride_w, int pad_h,
  //             int pad_w) {
  void im2col(new_addr_type addr) {
    int new_height = (H + 2 * pad_h - R) / stride_h + 1;
    assert(new_height == (int)P);
    int new_width = (W + 2 * pad_w - S) / stride_w + 1;
    assert(new_width == (int)Q);

    // new_addr_type *output =
    //     new new_addr_type[N * new_height * new_width * C * R * S];
    // new_addr_type *input = new new_addr_type[N * C * H * W];
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
                         h * new_width * C * R * S +
                         w * C * R * S + c * R * S +
                         i * S + j;
                if (h_pad >= 0 && h_pad < (int)H && w_pad >= 0 && w_pad < (int)W) {
                  output[out_index] =
                      input[n * C * H * W + c * H * W + h_pad * W + w_pad];
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


  std::string name;
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
  unsigned mapped;
  unsigned data_size;
  pim_layer_type type;
  unsigned pending_next;
  std::set<pim_layer *> issued_next;
  bool active;
  std::vector<pim_layer *> prev_layers;
  std::vector<pim_layer *> next_layers;
  std::vector<new_addr_type> matmul_addr;
  new_addr_type input_addr;
  unsigned input_size;
  new_addr_type output_addr;
  unsigned output_size;
  std::queue<warp_inst_t *> inst_queue;
  std::vector<pim_xbar *> m_xbars;
};

class pim_core_ctx : public core_t {
 public:
  // creator:
  pim_core_ctx(class gpgpu_sim *gpu, class pim_core_cluster *cluster,
               unsigned shader_id, unsigned tpc_id,
               const shader_core_config *config,
               const memory_config *mem_config, shader_core_stats *stats,
               pim_core_config *pim_config, pim_core_stats *pim_stats);

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

  void ldst_cycle();
  void memory_cycle();
  void issue();
  void control_cycle();
  bool issue_mem(warp_inst_t *inst);
  void execute();
  void commit();
  int test_res_bus(int latency);
  void create_exec_pipeline();
  Scoreboard *m_scoreboard;

  static const unsigned MAX_ALU_LATENCY = 512;
  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;
  pim_core_config *get_pim_core_config() { return m_pim_core_config; }
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_cache_stats(cache_stats &cs);
 protected:
  pim_core_config *m_pim_core_config;

 public:
  void inc_simt_to_mem(unsigned n_flits) {
    m_stats->n_simt_to_mem[m_sid] += n_flits;
  }
  PowerscalingCoefficients *scaling_coeffs;
  bool response_buffer_full() const;
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
  void map_layer_conv2d(pim_layer *layer);
  bool can_issue_layer(pim_layer *layer);

  pim_xbar *next_avail_xbar();

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // general information
  unsigned m_sid;  // shader id
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  class pim_core_cluster *m_cluster;
  std::vector<register_set *> m_issue_reg;
  std::vector<register_set *> m_result_reg;
  std::vector<register_set *> m_ldst_reg;
  std::unordered_map<warp_inst_t *, std::vector<new_addr_type>> m_addr_map;

  // statistics
  shader_core_stats *m_stats;
  pim_core_stats *m_pim_stats;

  // interconnect interface
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

  unsigned mf_size = 32;
  std::vector<pim_layer *> m_running_layers;
//  private:
  unsigned sent_bytes;
  unsigned used_xbars;
  bool core_full;
  
  simple_ldst_unit *m_ldst_unit;
  std::list<mem_fetch *> m_response_fifo;
  unsigned m_pending_loads;
  std::unordered_map<mem_fetch *, unsigned> m_loads;
  std::vector<pim_xbar *> m_xbars;
  unsigned last_checked_xbar;
  std::vector<scratchpad *> m_scratchpads;
  // new_addr_type weight_addr = 0xffff7fffffffffff;
  // new_addr_type input_addr =  0xffff8fffffffffff;
};

class pim_core_cluster : public simt_core_cluster {
 public:
  pim_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                   const shader_core_config *config,
                   const memory_config *mem_config, shader_core_stats *stats,
                   memory_stats_t *mstats, pim_core_config *pim_config, pim_core_stats *pim_stats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    full = false;
    m_pim_config = pim_config;
    m_pim_stats = pim_stats;
  };

  void core_cycle();
  void icnt_cycle();
  void ldst_cycle();
  void core_control_cycle();
  bool map_layer(pim_layer *layer);
  void get_cache_stats(cache_stats &cs) const;

  // void reinit();
  // void cache_flush();
  // void cache_invalidate();
  // bool icnt_injection_buffer_full(unsigned size, bool write);
  // void icnt_inject_request_packet(class mem_fetch *mf);

  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  // // for perfect memory interface
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  }
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf);
  }
  // virtual void create_pim_core_ctx() = 0;
  bool full;

//  protected:
//   unsigned m_cluster_id;
//   const shader_core_config *m_config;
//   shader_core_stats *m_stats;
//   memory_stats_t *m_memory_stats;
  pim_core_ctx **m_core;
  std::list<mem_fetch *> m_response_fifo;

  pim_core_config *m_pim_config;
  pim_core_stats *m_pim_stats;
};


class pim_core_config {
 public:
  pim_core_config(gpgpu_context *ctx) {
    num_xbars = 256;
    xbar_size_x = 256;
    xbar_size_y = 256;
    adc_count = 2;
    adc_precision = 4;
    dac_precision = 1;
    row_activation_rate = 4;

    program_latency = 5;
    integrate_latency = 5;
    sample_latency = 5;

    device_precision = 4;

    num_scratchpads = 4;
    scratchpad_size = 1024;
  
    gpgpu_ctx = ctx; 
    data_type = INT8_TYPE;
  }

  void init() {}
  // void reg_options(class OptionParser *opp);
  unsigned byte_per_row();
  unsigned get_data_size_byte();
  unsigned get_data_size_bit();
  unsigned num_device_per_weight();

  gpgpu_context *gpgpu_ctx;

  unsigned num_xbars;
  unsigned xbar_size_x;
  unsigned xbar_size_y;
  unsigned adc_count;
  unsigned program_latency;
  unsigned integrate_latency;
  unsigned sample_latency;
  pim_data_type data_type;
  unsigned device_precision;
  unsigned adc_precision;
  unsigned dac_precision;
  unsigned row_activation_rate;
  unsigned num_scratchpads;
  unsigned scratchpad_size;

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

// class xbar_memory_interface : public mem_fetch_interface {
//  public:
//   xbar_memory_interface(pim_core_ctx *core, pim_core_cluster *cluster, pim_xbar *xbar) {
//     m_core = core;
//     m_cluster = cluster;
//     m_xbar = xbar;
//   }
//   virtual bool full(unsigned size, bool write) const;
//   virtual void push(mem_fetch *mf);

//  private:
//   pim_core_ctx *m_core;
//   pim_core_cluster *m_cluster;
//   pim_xbar *m_xbar;
// };

// class pseudo_xbar_memory_interface : public xbar_memory_interface {
//  public:
//   pseudo_xbar_memory_interface(pim_core_ctx *core, pim_core_cluster *cluster,
//                                pim_xbar *xbar)
//       : xbar_memory_interface(core, cluster, xbar) {}
//   virtual bool full(unsigned size, bool write) const {
//     return false;
//   }
//   virtual void push(mem_fetch *mf) {

//   }
// };

class pim_core_stats : public shader_core_stats {
 public:
  pim_core_stats(const pim_core_config *pim_config,
                 const shader_core_config *shader_config)
      : shader_core_stats(shader_config) {
    m_pim_config = pim_config;
    input_loads_sent.resize(shader_config->n_pim_clusters, std::vector<unsigned>());
    xbar_device_used.resize(shader_config->n_pim_clusters, std::vector<unsigned>());
    xbar_integrate_count.resize(shader_config->n_pim_clusters, std::vector<unsigned>());
    xbar_program_cycle.resize(shader_config->n_pim_clusters, std::vector<unsigned>());
    xbar_sample_cycle.resize(shader_config->n_pim_clusters, std::vector<unsigned>());
    xbar_active_cycle.resize(shader_config->n_pim_clusters, std::vector<unsigned>());
    xbar_program_efficiency.resize(shader_config->n_pim_clusters, std::vector<unsigned>());
    xbar_executed_inst.resize(shader_config->n_pim_clusters, std::vector<unsigned>());
    for (unsigned i = 0; i < shader_config->n_pim_clusters; i++) {
      input_loads_sent[i].resize(pim_config->num_xbars, 0);
      xbar_device_used[i].resize(pim_config->num_xbars, 0);
      xbar_integrate_count[i].resize(pim_config->num_xbars, 0);
      xbar_program_cycle[i].resize(pim_config->num_xbars, 0);
      xbar_sample_cycle[i].resize(pim_config->num_xbars, 0);
      xbar_active_cycle[i].resize(pim_config->num_xbars, 0);
      xbar_program_efficiency[i].resize(pim_config->num_xbars, 0);
      xbar_executed_inst[i].resize(pim_config->num_xbars, 0);
    }
  }

  void print(FILE *fout, unsigned long long tot_cycle) const;

  std::vector<std::vector<unsigned>> input_loads_sent;
  std::vector<std::vector<unsigned>> xbar_device_used;
  std::vector<std::vector<unsigned>> xbar_integrate_count;
  std::vector<std::vector<unsigned>> xbar_program_cycle;
  std::vector<std::vector<unsigned>> xbar_sample_cycle;
  std::vector<std::vector<unsigned>> xbar_active_cycle;
  std::vector<std::vector<unsigned>> xbar_program_efficiency;
  std::vector<std::vector<unsigned>> xbar_executed_inst;

  const pim_core_config *m_pim_config;

  friend class gpgpu_sim;
  friend class pim_core_ctx;
};

#define PROGRAM_REG 1
#define COMPUTE_REG 2
#define SAMPLE_REG 3
class pim_xbar : public simd_function_unit {
 public:
  pim_xbar(register_set *result_port, const shader_core_config *config,
          pim_core_ctx *core, unsigned issue_reg_id)
      : simd_function_unit(config) {
    m_result_port = result_port;
    m_pipeline_depth = 4; //one for each op, one for result
    m_pipeline_reg = new warp_inst_t *[m_pipeline_depth];
    for (unsigned i = 0; i < m_pipeline_depth; i++) {
      m_pipeline_reg[i] = new warp_inst_t(config);
    }
    m_core = core;
    m_issue_reg_id = issue_reg_id;

    m_name = "XBAR";
    m_status = XBAR_INITIATED;
    m_xbar_id = issue_reg_id;
    used_rows = 0;
    used_cols = 0;
    byte_per_row = 0;
    sent_bytes = 0;
    programmed_rows = 0;
    done_activation = 0;
    total_activation = 0;
    sample_scale_factor = 1;

    sampling = false;
    computing = false;
    programming = false;

    mapped = false;
    active = false;
    stall = true;

    m_program_latency = 0;
    m_compute_latency = 0;
    m_sample_latency = 0;
    executed_inst = 0;
    op_id = 0;

    m_core = core;
    m_pim_config = core->get_pim_core_config();
    m_gpu = m_core->get_gpu();
    weight = buffer();

    // m_xbar_interface = new pseudo_xbar_memory_interface(m_core, m_core->m_cluster, this);
  }


  virtual bool can_issue(const warp_inst_t &inst) const {
    if (!m_dispatch_reg->empty()) {
      return false;
    }
    // dispatch reg is free

    // if (stall) {
    //   return false;
    // }

    if (programming) {
      // during programming, cannot do anything else
      return false;
    }
    
    if (inst.op == XBAR_SAMPLE_OP) {
      return !(sampling || computing);
    } else if (inst.op == XBAR_INTEGRATE_OP) {
      return !computing;
    } 

    if (inst.op == EXIT_OPS) {
      return !(sampling || computing || programming);
    }

    return true;
  }

  virtual void cycle() {
    if (!m_dispatch_reg->empty()) {
      if (m_dispatch_reg->op == XBAR_PROGRAM_OP) {
        if (m_pipeline_reg[PROGRAM_REG]->empty()) {
          m_program_latency = m_dispatch_reg->latency;
          move_warp(m_pipeline_reg[PROGRAM_REG], m_dispatch_reg);
        }
      } else if (m_dispatch_reg->op == XBAR_INTEGRATE_OP) {
        if (m_pipeline_reg[COMPUTE_REG]->empty()) {
          m_core->m_pim_stats
              ->xbar_integrate_count[m_core->m_tpc - m_config->n_simt_clusters]
                                    [m_xbar_id]++;
          m_compute_latency = m_dispatch_reg->latency;
          move_warp(m_pipeline_reg[COMPUTE_REG], m_dispatch_reg);
        }
      } else if (m_dispatch_reg->op == XBAR_SAMPLE_OP) {
        if (m_pipeline_reg[SAMPLE_REG]->empty()) {
          m_sample_latency = m_dispatch_reg->latency;
          move_warp(m_pipeline_reg[SAMPLE_REG], m_dispatch_reg);
        }
      } else if (m_dispatch_reg->op == EXIT_OPS) {
        if (m_pipeline_reg[0]->empty()) {
          move_warp(m_pipeline_reg[0], m_dispatch_reg);
        }
      }
    }
    
    bool cycled = false;
    if (!m_pipeline_reg[PROGRAM_REG]->empty()) {
      // extra caution
      assert(m_pipeline_reg[COMPUTE_REG]->empty() &&
             m_pipeline_reg[SAMPLE_REG]->empty() && m_dispatch_reg->empty());
      if (m_program_latency > 0) {
        m_program_latency--;
        cycled = true;
      } else {
        move_warp(m_pipeline_reg[0], m_pipeline_reg[PROGRAM_REG]);
      }
    }

    if (!m_pipeline_reg[COMPUTE_REG]->empty()) {
      assert(m_pipeline_reg[PROGRAM_REG]->empty());
      if (m_compute_latency > 0) {
        m_compute_latency--;
        cycled = true;
      } else {
        move_warp(m_pipeline_reg[0], m_pipeline_reg[COMPUTE_REG]);
      }
    }

    if (!m_pipeline_reg[SAMPLE_REG]->empty()) {
      assert(m_pipeline_reg[PROGRAM_REG]->empty());
      if (m_sample_latency > 0) {
        m_sample_latency--;
        cycled = true;
      } else {
        move_warp(m_pipeline_reg[0], m_pipeline_reg[SAMPLE_REG]);
      }
    }

    if (cycled) {
      m_core->m_pim_stats
          ->xbar_active_cycle[m_core->m_tpc - m_config->n_simt_clusters]
                             [m_xbar_id]++;
    }

    if (!m_pipeline_reg[0]->empty()) {
      if (m_pipeline_reg[0]->op == XBAR_SAMPLE_OP) {
        sampling = false;
      } else if (m_pipeline_reg[0]->op == XBAR_INTEGRATE_OP) {
        computing = false;
      } else if (m_pipeline_reg[0]->op == XBAR_PROGRAM_OP) {
        programming = false;
      }

      m_result_port->move_in(m_pipeline_reg[0]);
    }
  }
  virtual unsigned get_issue_reg_id() { return m_issue_reg_id; }
  virtual bool stallable() const { return false; }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return false; }
  // bool xbar_icnt_injection_buffer_full(unsigned size, bool write);
  // void xbar_icnt_inject_request_packet(mem_fetch *mf);
  unsigned get_used_devices() {
    return used_rows * used_cols;
  }

  unsigned m_pipeline_depth;
  warp_inst_t **m_pipeline_reg;
  register_set *m_result_port;
  unsigned m_issue_reg_id;
  unsigned m_program_latency;
  unsigned m_compute_latency;
  unsigned m_sample_latency;

  xbar_status m_status;
  unsigned used_rows, used_cols;
  unsigned m_xbar_id;
  unsigned byte_per_row;
  unsigned sent_bytes;
  unsigned programmed_rows;
  unsigned done_activation;
  unsigned total_activation;
  unsigned sample_scale_factor;
  unsigned executed_inst;
  unsigned op_id;
  unsigned xbar_row_id;
  unsigned xbar_col_id;
  pim_layer *m_layer;
  pim_core_ctx *m_core;
  gpgpu_sim *m_gpu;
  pim_core_config *m_pim_config;
  // xbar_memory_interface *m_xbar_interface;
  buffer weight;
  std::vector<unsigned> m_op_pending_loads;
  std::queue<new_addr_type> m_load_queue;
  std::unordered_map<new_addr_type, std::set<unsigned>> addr_to_op;
  std::queue<warp_inst_t *> inst_queue;

  std::deque<new_addr_type> regs_order;
  std::set<new_addr_type> regs_value;

  bool sampling;
  bool computing;
  bool programming;
  bool mapped;
  bool active;
  bool stall;
};


class controller {
  public:
  controller(pim_core_ctx *core, pim_core_cluster *cluster, pim_core_config *pim_config, pim_core_stats *pim_stats) {
    m_core = core;
    m_cluster = cluster;
    m_pim_config = pim_config;
    m_pim_stats = pim_stats;
  }

  pim_core_ctx *m_core;
  pim_core_cluster *m_cluster;
  pim_core_config *m_pim_config;
  pim_core_stats *m_pim_stats;
};

class simple_ldst_unit : public ldst_unit {
 public:
  simple_ldst_unit(mem_fetch_interface *icnt,
                   shader_core_mem_fetch_allocator *mf_allocator,
                   const shader_core_config *config,
                   const memory_config *mem_config,
                   class shader_core_stats *stats, unsigned sid, unsigned tpc,
                   gpgpu_sim *gpu, exec_shader_core_ctx *core, pim_core_ctx *pim_core, Scoreboard *scoreboard)
      : ldst_unit(icnt, mf_allocator, core, NULL, NULL, config, mem_config,
                  stats, sid, tpc) {
    m_pim_core = pim_core;
    m_scoreboard = scoreboard;
  }

  void cycle();
  void issue(register_set &reg_set);
  void writeback();
  void L1_latency_queue_cycle();

  // gpgpu_sim *m_gpu;
  pim_core_ctx *m_pim_core;
};

#endif
