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
    : core_t(gpu, NULL, config->warp_size, config->n_thread_per_shader),
      m_barriers(NULL, config->max_warps_per_shader, config->max_cta_per_core,
                 config->max_barriers_per_cta, config->warp_size),
      m_active_warps(0),
      m_dynamic_warp_id(0) {
  //   m_cluster = cluster;
  //   m_config = config;
  //   m_memory_config = mem_config;
  //   m_stats = stats;
  //   unsigned warp_size = config->warp_size;
  //   Issue_Prio = 0;

  //   m_sid = shader_id;
  //   m_tpc = tpc_id;

  //   if(get_gpu()->get_config().g_power_simulation_enabled){
  //     scaling_coeffs =  get_gpu()->get_scaling_coeffs();
  //   }

  //   m_last_inst_gpu_sim_cycle = 0;
  //   m_last_inst_gpu_tot_sim_cycle = 0;

  //   // Jin: for concurrent kernels on a SM
  //   m_occupied_n_threads = 0;
  //   m_occupied_shmem = 0;
  //   m_occupied_regs = 0;
  //   m_occupied_ctas = 0;
  //   m_occupied_hwtid.reset();
  //   m_occupied_cta_to_hwtid.clear();
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

pim_core_cluster::pim_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                                   const shader_core_config *config,
                                   const memory_config *mem_config,
                                   shader_core_stats *stats,
                                   class memory_stats_t *mstats) {
  m_config = config;
  m_cta_issue_next_core = m_config->n_simt_cores_per_cluster -
                          1;  // this causes first launch to use hw cta 0
  m_cluster_id = cluster_id;
  m_gpu = gpu;
  m_stats = stats;
  m_memory_stats = mstats;
  m_mem_config = mem_config;
}