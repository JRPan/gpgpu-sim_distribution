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

class pim_core_ctx : public core_t {
 public:
  // creator:
  pim_core_ctx(class gpgpu_sim *gpu, class pim_core_cluster *cluster,
                  unsigned shader_id, unsigned tpc_id,
                  const shader_core_config *config,
                  const memory_config *mem_config, shader_core_stats *stats);

  // used by pim_core_cluster:
  // modifiers
//   void cycle();
//   void reinit(unsigned start_thread, unsigned end_thread,
//               bool reset_not_completed);
//   void issue_block2core(class kernel_info_t &kernel);

//   void cache_flush();
//   void cache_invalidate();
//   void accept_fetch_response(mem_fetch *mf);
//   void accept_ldst_unit_response(class mem_fetch *mf);
//   void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
//                                    warp_set_t warps);
//   void set_kernel(kernel_info_t *k) {
//     assert(k);
//     m_kernel = k;
//     //        k->inc_running();
//     printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid,
//            m_kernel->get_uid(), m_kernel->name().c_str());
//   }
//   PowerscalingCoefficients *scaling_coeffs;
//   // accessors
//   bool fetch_unit_response_buffer_full() const;
//   bool ldst_unit_response_buffer_full() const;
//   unsigned get_not_completed() const { return m_not_completed; }
//   unsigned get_n_active_cta() const { return m_n_active_cta; }
//   unsigned isactive() const {
//     if (m_n_active_cta > 0)
//       return 1;
//     else
//       return 0;
//   }
//   kernel_info_t *get_kernel() { return m_kernel; }
//   unsigned get_sid() const { return m_sid; }

//   // used by functional simulation:
//   // modifiers
  virtual void warp_exit(unsigned warp_id);

//   // accessors
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const;
//   void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const;
//   float get_current_occupancy(unsigned long long &active,
//                               unsigned long long &total) const;

//   // used by pipeline timing model components:
//   // modifiers
//   void mem_instruction_stats(const warp_inst_t &inst);
//   void decrement_atomic_count(unsigned wid, unsigned n);
//   void inc_store_req(unsigned warp_id) { m_warp[warp_id]->inc_store_req(); }
//   void dec_inst_in_pipeline(unsigned warp_id) {
//     m_warp[warp_id]->dec_inst_in_pipeline();
//   }  // also used in writeback()
//   void store_ack(class mem_fetch *mf);
//   bool warp_waiting_at_mem_barrier(unsigned warp_id);
//   void set_max_cta(const kernel_info_t &kernel);
//   void warp_inst_complete(const warp_inst_t &inst);

//   // accessors
//   std::list<unsigned> get_regs_written(const inst_t &fvt) const;
//   const shader_core_config *get_config() const { return m_config; }
//   void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
//                          unsigned &dl1_misses);

//   void get_cache_stats(cache_stats &cs);
//   void get_L1I_sub_stats(unsigned kernel_id, struct cache_sub_stats &css) const;
//   void get_L1D_sub_stats(unsigned kernel_id, struct cache_sub_stats &css) const;
//   void get_L1C_sub_stats(unsigned kernel_id, struct cache_sub_stats &css) const;
//   void get_L1T_sub_stats(unsigned kernel_id, struct cache_sub_stats &css) const;
//   void update_cache_stats_size(unsigned kernel_id) {
//     m_L1I->update_stats_size(kernel_id);
//     m_ldst_unit->update_cache_stats_size(kernel_id);
//   }

//   void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

//   // debug:
//   void display_simt_state(FILE *fout, int mask) const;
//   void display_pipeline(FILE *fout, int print_mem, int mask3bit) const;

//   void incload_stat() { m_stats->m_num_loadqueued_insn[m_sid]++; }
//   void incstore_stat() { m_stats->m_num_storequeued_insn[m_sid]++; }
//   void incialu_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_nonsfu(active_count, latency);
//     }else {
//       m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//   }
//   void incimul_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_nonsfu(active_count, latency);
//     }else {
//       m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//   }
//   void incimul24_stat(unsigned active_count,double latency) {
//   if(m_config->gpgpu_clock_gated_lanes==false){
//     m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_nonsfu(active_count, latency);
//     }else {
//       m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;    
//    }
//    void incimul32_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+(double)active_count*latency
//          + inactive_lanes_accesses_sfu(active_count, latency);          
//     }else{
//       m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//   }
//    void incidiv_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+(double)active_count*latency
//          + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else {
//       m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;    
//   }
//    void incfpalu_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+(double)active_count*latency
//          + inactive_lanes_accesses_nonsfu(active_count, latency);
//     }else {
//     m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;     
//   }
//    void incfpmul_stat(unsigned active_count,double latency) {
//               // printf("FP MUL stat increament\n");
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_nonsfu(active_count, latency);
//     }else {
//     m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//    }
//    void incfpdiv_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else {
//       m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//    }
//    void incdpalu_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_dp_acesses[m_sid]=m_stats->m_num_dp_acesses[m_sid]+(double)active_count*latency
//          + inactive_lanes_accesses_nonsfu(active_count, latency);
//     }else {
//     m_stats->m_num_dp_acesses[m_sid]=m_stats->m_num_dp_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++; 
//    }
//    void incdpmul_stat(unsigned active_count,double latency) {
//               // printf("FP MUL stat increament\n");
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_dpmul_acesses[m_sid]=m_stats->m_num_dpmul_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_nonsfu(active_count, latency);
//     }else {
//     m_stats->m_num_dpmul_acesses[m_sid]=m_stats->m_num_dpmul_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//    }
//    void incdpdiv_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_dpdiv_acesses[m_sid]=m_stats->m_num_dpdiv_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else {
//       m_stats->m_num_dpdiv_acesses[m_sid]=m_stats->m_num_dpdiv_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//    }

//    void incsqrt_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_sqrt_acesses[m_sid]=m_stats->m_num_sqrt_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else{
//       m_stats->m_num_sqrt_acesses[m_sid]=m_stats->m_num_sqrt_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//    }

//    void inclog_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_log_acesses[m_sid]=m_stats->m_num_log_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else{
//       m_stats->m_num_log_acesses[m_sid]=m_stats->m_num_log_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//    }

//    void incexp_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_exp_acesses[m_sid]=m_stats->m_num_exp_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else{
//       m_stats->m_num_exp_acesses[m_sid]=m_stats->m_num_exp_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//   }

//    void incsin_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_sin_acesses[m_sid]=m_stats->m_num_sin_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else{
//       m_stats->m_num_sin_acesses[m_sid]=m_stats->m_num_sin_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//   }


//    void inctensor_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_tensor_core_acesses[m_sid]=m_stats->m_num_tensor_core_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else{
//       m_stats->m_num_tensor_core_acesses[m_sid]=m_stats->m_num_tensor_core_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//   }

//   void inctex_stat(unsigned active_count,double latency) {
//     if(m_config->gpgpu_clock_gated_lanes==false){
//       m_stats->m_num_tex_acesses[m_sid]=m_stats->m_num_tex_acesses[m_sid]+(double)active_count*latency
//         + inactive_lanes_accesses_sfu(active_count, latency); 
//     }else{
//       m_stats->m_num_tex_acesses[m_sid]=m_stats->m_num_tex_acesses[m_sid]+(double)active_count*latency;
//     }
//     m_stats->m_active_exu_threads[m_sid]+=active_count;
//     m_stats->m_active_exu_warps[m_sid]++;
//   }

//   void inc_const_accesses(unsigned active_count) {
//     m_stats->m_num_const_acesses[m_sid]=m_stats->m_num_const_acesses[m_sid]+active_count;
//   }

//   void incsfu_stat(unsigned active_count, double latency) {
//     m_stats->m_num_sfu_acesses[m_sid] =
//         m_stats->m_num_sfu_acesses[m_sid] + (double)active_count*latency;
//   }
//   void incsp_stat(unsigned active_count, double latency) {
//     m_stats->m_num_sp_acesses[m_sid] =
//         m_stats->m_num_sp_acesses[m_sid] + (double)active_count*latency;
//   }
//   void incmem_stat(unsigned active_count, double latency) {
//     if (m_config->gpgpu_clock_gated_lanes == false) {
//       m_stats->m_num_mem_acesses[m_sid] =
//           m_stats->m_num_mem_acesses[m_sid] + (double)active_count*latency +
//           inactive_lanes_accesses_nonsfu(active_count, latency);
//     } else {
//       m_stats->m_num_mem_acesses[m_sid] =
//           m_stats->m_num_mem_acesses[m_sid] + (double)active_count*latency;
//     }
//   }
//   void incexecstat(warp_inst_t *&inst);

//   void incregfile_reads(unsigned active_count) {
//     m_stats->m_read_regfile_acesses[m_sid] =
//         m_stats->m_read_regfile_acesses[m_sid] + active_count;
//   }
//   void incregfile_writes(unsigned active_count) {
//     m_stats->m_write_regfile_acesses[m_sid] =
//         m_stats->m_write_regfile_acesses[m_sid] + active_count;
//   }
//   void incnon_rf_operands(unsigned active_count) {
//     m_stats->m_non_rf_operands[m_sid] =
//         m_stats->m_non_rf_operands[m_sid] + active_count;
//   }

//   void incspactivelanes_stat(unsigned active_count) {
//     m_stats->m_active_sp_lanes[m_sid] =
//         m_stats->m_active_sp_lanes[m_sid] + active_count;
//   }
//   void incsfuactivelanes_stat(unsigned active_count) {
//     m_stats->m_active_sfu_lanes[m_sid] =
//         m_stats->m_active_sfu_lanes[m_sid] + active_count;
//   }
//   void incfuactivelanes_stat(unsigned active_count) {
//     m_stats->m_active_fu_lanes[m_sid] =
//         m_stats->m_active_fu_lanes[m_sid] + active_count;
//   }
//   void incfumemactivelanes_stat(unsigned active_count) {
//     m_stats->m_active_fu_mem_lanes[m_sid] =
//         m_stats->m_active_fu_mem_lanes[m_sid] + active_count;
//   }

//   void inc_simt_to_mem(unsigned n_flits) {
//     m_stats->n_simt_to_mem[m_sid] += n_flits;
//   }
//   bool check_if_non_released_reduction_barrier(warp_inst_t &inst);

//  protected:
//   unsigned inactive_lanes_accesses_sfu(unsigned active_count, double latency) {
//     return (((32 - active_count) >> 1) * latency) +
//            (((32 - active_count) >> 3) * latency) +
//            (((32 - active_count) >> 3) * latency);
//   }
//   unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,
//                                           double latency) {
//     return (((32 - active_count) >> 1) * latency);
//   }

//   int test_res_bus(int latency);
//   address_type next_pc(int tid) const;
//   void fetch();
//   void register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel);

//   void decode();

//   void issue();
//   friend class scheduler_unit;  // this is needed to use private issue warp.
//   friend class TwoLevelScheduler;
//   friend class LooseRoundRobbinScheduler;
  virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);

//   void create_front_pipeline();
//   void create_schedulers();
//   void create_exec_pipeline();

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

//   // Returns numbers of addresses in translated_addrs
//   unsigned translate_local_memaddr(address_type localaddr, unsigned tid,
//                                    unsigned num_shader, unsigned datasize,
//                                    new_addr_type *translated_addrs);

//   void read_operands();

//   void execute();

//   void writeback();

//   // used in display_pipeline():
//   void dump_warp_state(FILE *fout) const;
//   void print_stage(unsigned int stage, FILE *fout) const;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // general information
  unsigned m_sid;  // shader id
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  class pim_core_cluster *m_cluster;

  // statistics
  shader_core_stats *m_stats;

  // CTA scheduling / hardware thread allocation
  unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                            // currently running on this shader.
  unsigned m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  unsigned m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;

  // thread contexts
  thread_ctx_t *m_threadState;

  // interconnect interface
  mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

  // fetch
  read_only_cache *m_L1I;  // instruction cache
  int m_last_warp_fetched;

  // decode/dispatch
  std::vector<shd_warp_t *> m_warp;  // per warp information array
  barrier_set_t m_barriers;
  ifetch_buffer_t m_inst_fetch_buffer;
  std::vector<register_set> m_pipeline_reg;
  Scoreboard *m_scoreboard;
  opndcoll_rfu_t m_operand_collector;
  int m_active_warps;
  std::vector<register_set *> m_specilized_dispatch_reg;

  // schedule
  std::vector<scheduler_unit *> schedulers;

  // issue
  unsigned int Issue_Prio;

  // execute
  unsigned m_num_function_units;
  std::vector<unsigned> m_dispatch_port;
  std::vector<unsigned> m_issue_port;
  std::vector<simd_function_unit *>
      m_fu;  // stallable pipelines should be last in this array
  ldst_unit *m_ldst_unit;
  static const unsigned MAX_ALU_LATENCY = 512;
  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;

  // used for local address mapping with single kernel launch
  unsigned kernel_max_cta_per_shader;
  unsigned kernel_padded_threads_per_cta;
  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned m_dynamic_warp_id;

  // Jin: concurrent kernels on a sm
//  public:
//   bool can_issue_1block(kernel_info_t &kernel);
//   bool occupy_shader_resource_1block(kernel_info_t &kernel, bool occupy);
//   void release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &kernel);
//   int find_available_hwtid(unsigned int cta_size, bool occupy);

 private:
  unsigned int m_occupied_n_threads;
  unsigned int m_occupied_shmem;
  unsigned int m_occupied_regs;
  unsigned int m_occupied_ctas;
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;
};

class pim_core_cluster {
 public:
  pim_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                    const shader_core_config *config,
                    const memory_config *mem_config, shader_core_stats *stats,
                    memory_stats_t *mstats);

  // void core_cycle();
  // void icnt_cycle();

  // void reinit();
  // unsigned issue_block2core();
  // void cache_flush();
  // void cache_invalidate();
  // bool icnt_injection_buffer_full(unsigned size, bool write);
  // void icnt_inject_request_packet(class mem_fetch *mf);

  // // for perfect memory interface
  // bool response_queue_full() {
  //   return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  // }
  // void push_response_fifo(class mem_fetch *mf) {
  //   m_response_fifo.push_back(mf);
  // }

  // void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
  //                              unsigned *rpc) const;
  // unsigned max_cta(const kernel_info_t &kernel);
  // unsigned get_not_completed() const;
  // void print_not_completed(FILE *fp) const;
  // unsigned get_n_active_cta() const;
  // unsigned get_n_active_sms() const;
  // gpgpu_sim *get_gpu() { return m_gpu; }

  // void display_pipeline(unsigned sid, FILE *fout, int print_mem, int mask);
  // void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
  //                        unsigned &dl1_misses) const;

  // void get_cache_stats(cache_stats &cs) const;
  // void get_L1I_sub_stats(unsigned kernel_id, struct cache_sub_stats &css) const;
  // void get_L1D_sub_stats(unsigned kernel_id, struct cache_sub_stats &css) const;
  // void get_L1C_sub_stats(unsigned kernel_id, struct cache_sub_stats &css) const;
  // void get_L1T_sub_stats(unsigned kernel_id, struct cache_sub_stats &css) const;
  // void update_cache_stats_size(unsigned kernel_id) {
  //   for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
  //     m_core[i]->update_cache_stats_size(kernel_id);
  //   }
  // }

  // void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;
  // float get_current_occupancy(unsigned long long &active,
  //                             unsigned long long &total) const;
  // virtual void create_pim_core_ctx() = 0;

 protected:
  unsigned m_cluster_id;
  gpgpu_sim *m_gpu;
  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  memory_stats_t *m_memory_stats;
  pim_core_ctx **m_core;
  const memory_config *m_mem_config;

  unsigned m_cta_issue_next_core;
  std::list<unsigned> m_core_sim_order;
  std::list<mem_fetch *> m_response_fifo;
};

#endif