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

#ifndef pim_icnt_WRAPPER_H
#define pim_icnt_WRAPPER_H

#include <stdio.h>

// functional interface to the interconnect

typedef void (*pim_icnt_create_p)(unsigned n_shader, unsigned n_mem);
typedef void (*pim_icnt_init_p)();
typedef bool (*pim_icnt_has_buffer_p)(unsigned input, unsigned int size);
typedef void (*pim_icnt_push_p)(unsigned input, unsigned output, void* data,
                            unsigned int size);
typedef void* (*pim_icnt_pop_p)(unsigned output);
typedef void (*pim_icnt_transfer_p)();
typedef bool (*pim_icnt_busy_p)();
typedef void (*pim_icnt_drain_p)();
typedef void (*pim_icnt_display_stats_p)();
typedef void (*pim_icnt_display_overall_stats_p)();
typedef void (*pim_icnt_display_state_p)(FILE* fp);
typedef unsigned (*pim_icnt_get_flit_size_p)();

extern pim_icnt_create_p pim_icnt_create;
extern pim_icnt_init_p pim_icnt_init;
extern pim_icnt_has_buffer_p pim_icnt_has_buffer;
extern pim_icnt_push_p pim_icnt_push;
extern pim_icnt_pop_p pim_icnt_pop;
extern pim_icnt_transfer_p pim_icnt_transfer;
extern pim_icnt_busy_p pim_icnt_busy;
extern pim_icnt_drain_p pim_icnt_drain;
extern pim_icnt_display_stats_p pim_icnt_display_stats;
extern pim_icnt_display_overall_stats_p pim_icnt_display_overall_stats;
extern pim_icnt_display_state_p pim_icnt_display_state;
extern pim_icnt_get_flit_size_p pim_icnt_get_flit_size;
extern unsigned g_pim_network_mode;

enum pim_network_mode { PIM_INTERSIM = 1, PIM_LOCAL_XBAR = 2, N_PIM_NETWORK_MODE };

void pim_icnt_wrapper_init();
void pim_icnt_reg_options(class OptionParser* opp);

#endif
