// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#include "pim_icnt_wrapper.h"
#include <assert.h>
#include "../intersim2/globals.hpp"
#include "../intersim2/interconnect_interface.hpp"
#include "local_interconnect.h"

pim_icnt_create_p pim_icnt_create;
pim_icnt_init_p pim_icnt_init;
pim_icnt_has_buffer_p pim_icnt_has_buffer;
pim_icnt_push_p pim_icnt_push;
pim_icnt_pop_p pim_icnt_pop;
pim_icnt_transfer_p pim_icnt_transfer;
pim_icnt_busy_p pim_icnt_busy;
pim_icnt_display_stats_p pim_icnt_display_stats;
pim_icnt_display_overall_stats_p pim_icnt_display_overall_stats;
pim_icnt_display_state_p pim_icnt_display_state;
pim_icnt_get_flit_size_p pim_icnt_get_flit_size;

unsigned g_pim_network_mode;
char* g_pim_network_config_filename;

struct inct_config g_pim_inct_config;
LocalInterconnect* g_pim_local_icnt_interface;
InterconnectInterface *g_pim_icnt_interface;

#include "../option_parser.h"

// Wrapper to intersim2 to accompany old pim_icnt_wrapper
// TODO: use delegate/boost/c++11<funtion> instead

static void pim_intersim2_create(unsigned int n_shader, unsigned int n_mem) {
  g_pim_icnt_interface->CreateInterconnect(n_shader, n_mem);
}

static void pim_intersim2_init() { g_pim_icnt_interface->Init(); }

static bool pim_intersim2_has_buffer(unsigned input, unsigned int size) {
  return g_pim_icnt_interface->HasBuffer(input, size);
}

static void pim_intersim2_push(unsigned input, unsigned output, void* data,
                           unsigned int size) {
  g_pim_icnt_interface->Push(input, output, data, size);
}

static void* pim_intersim2_pop(unsigned output) {
  return g_pim_icnt_interface->Pop(output);
}

static void pim_intersim2_transfer() { g_pim_icnt_interface->Advance(); }

static bool pim_intersim2_busy() { return g_pim_icnt_interface->Busy(); }

static void pim_intersim2_display_stats() { g_pim_icnt_interface->DisplayStats(); }

static void pim_intersim2_display_overall_stats() {
  g_pim_icnt_interface->DisplayOverallStats();
}

static void pim_intersim2_display_state(FILE* fp) {
  g_pim_icnt_interface->DisplayState(fp);
}

static unsigned pim_intersim2_get_flit_size() {
  return g_pim_icnt_interface->GetFlitSize();
}

//////////////////////////////////////////////////////

static void pim_LocalInterconnect_create(unsigned int n_shader,
                                     unsigned int n_mem) {
  g_pim_local_icnt_interface->CreateInterconnect(n_shader, n_mem);
}

static void pim_LocalInterconnect_init() { g_pim_local_icnt_interface->Init(); }

static bool pim_LocalInterconnect_has_buffer(unsigned input, unsigned int size) {
  return g_pim_local_icnt_interface->HasBuffer(input, size);
}

static void pim_LocalInterconnect_push(unsigned input, unsigned output, void* data,
                                   unsigned int size) {
  g_pim_local_icnt_interface->Push(input, output, data, size);
}

static void* pim_LocalInterconnect_pop(unsigned output) {
  return g_pim_local_icnt_interface->Pop(output);
}

static void pim_LocalInterconnect_transfer() { g_pim_local_icnt_interface->Advance(); }

static bool pim_LocalInterconnect_busy() { return g_pim_local_icnt_interface->Busy(); }

static void pim_LocalInterconnect_display_stats() {
  g_pim_local_icnt_interface->DisplayStats();
}

static void pim_LocalInterconnect_display_overall_stats() {
  g_pim_local_icnt_interface->DisplayOverallStats();
}

static void pim_LocalInterconnect_display_state(FILE* fp) {
  g_pim_local_icnt_interface->DisplayState(fp);
}

static unsigned pim_LocalInterconnect_get_flit_size() {
  return g_pim_local_icnt_interface->GetFlitSize();
}

///////////////////////////

void pim_icnt_reg_options(class OptionParser* opp) {
  option_parser_register(opp, "-pim_network_mode", OPT_INT32, &g_pim_network_mode,
                         "Interconnection network mode", "2");
  option_parser_register(opp, "-pim_inter_config_file", OPT_CSTR,
                         &g_pim_network_config_filename,
                         "Interconnection network config file", "mesh");

  // parameters for local xbar
  option_parser_register(opp, "-pim_icnt_in_buffer_limit", OPT_UINT32,
                         &g_pim_inct_config.in_buffer_limit, "in_buffer_limit",
                         "64");
  option_parser_register(opp, "-pim_icnt_out_buffer_limit", OPT_UINT32,
                         &g_pim_inct_config.out_buffer_limit, "out_buffer_limit",
                         "64");
  option_parser_register(opp, "-pim_icnt_subnets", OPT_UINT32,
                         &g_pim_inct_config.subnets, "subnets", "2");
  option_parser_register(opp, "-pim_icnt_arbiter_algo", OPT_UINT32,
                         &g_pim_inct_config.arbiter_algo, "arbiter_algo", "1");
  option_parser_register(opp, "-pim_icnt_verbose", OPT_UINT32,
                         &g_pim_inct_config.verbose, "inct_verbose", "0");
  option_parser_register(opp, "-pim_icnt_grant_cycles", OPT_UINT32,
                         &g_pim_inct_config.grant_cycles, "grant_cycles", "1");
}

void pim_icnt_wrapper_init() {
  switch (g_pim_network_mode) {
    case PIM_INTERSIM:
      // FIXME: delete the object: may add pim_icnt_done wrapper
      g_pim_icnt_interface = InterconnectInterface::New(g_pim_network_config_filename);
      pim_icnt_create = pim_intersim2_create;
      pim_icnt_init = pim_intersim2_init;
      pim_icnt_has_buffer = pim_intersim2_has_buffer;
      pim_icnt_push = pim_intersim2_push;
      pim_icnt_pop = pim_intersim2_pop;
      pim_icnt_transfer = pim_intersim2_transfer;
      pim_icnt_busy = pim_intersim2_busy;
      pim_icnt_display_stats = pim_intersim2_display_stats;
      pim_icnt_display_overall_stats = pim_intersim2_display_overall_stats;
      pim_icnt_display_state = pim_intersim2_display_state;
      pim_icnt_get_flit_size = pim_intersim2_get_flit_size;
      break;
    case PIM_LOCAL_XBAR:
      g_pim_local_icnt_interface = LocalInterconnect::New(g_pim_inct_config);
      pim_icnt_create = pim_LocalInterconnect_create;
      pim_icnt_init = pim_LocalInterconnect_init;
      pim_icnt_has_buffer = pim_LocalInterconnect_has_buffer;
      pim_icnt_push = pim_LocalInterconnect_push;
      pim_icnt_pop = pim_LocalInterconnect_pop;
      pim_icnt_transfer = pim_LocalInterconnect_transfer;
      pim_icnt_busy = pim_LocalInterconnect_busy;
      pim_icnt_display_stats = pim_LocalInterconnect_display_stats;
      pim_icnt_display_overall_stats = pim_LocalInterconnect_display_overall_stats;
      pim_icnt_display_state = pim_LocalInterconnect_display_state;
      pim_icnt_get_flit_size = pim_LocalInterconnect_get_flit_size;
      break;
    default:
      assert(0);
      break;
  }
}
