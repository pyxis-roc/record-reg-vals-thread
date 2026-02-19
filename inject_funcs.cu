/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>
#include <cstdarg>

#include "utils/utils.h"

/* for channel */
#define USE_ASYNC_STREAM
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

__device__ bool warp_selected(uint64_t pselected, uint32_t cta_x, uint32_t cta_y, uint32_t cta_z, uint32_t warp_id) {
  warp_selection_t *wst = (warp_selection_t *) pselected;

  return wst->cta_x == cta_x && wst->cta_y == cta_y && wst->cta_z == cta_z && wst->warp_id == warp_id;
}


extern "C" __device__ __noinline__ void record_reg_val_thread(int pred, int opcode_id,
                                                              int opcode_idx,
                                                              uint64_t pchannel_dev,
							      uint64_t pselected,
							      int32_t constant,
							      int32_t pred_regs,
							      int32_t upred_regs,
                                                              int32_t num_regs,
							      int32_t unum_regs...) {
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    reg_info_t ri;

    int4 cta = get_ctaid();
    int warp_id = get_warpid();

    if(!warp_selected(pselected, cta.x, cta.y, cta.z, warp_id))
      return;
    ri.cta_id_x = cta.x;
    ri.cta_id_y = cta.y;
    ri.cta_id_z = cta.z;
    ri.warp_id = warp_id;

    ri.opcode_idx = opcode_idx;
    ri.opcode_id = opcode_id;
    ri.num_regs = num_regs;
    ri.unum_regs = unum_regs;
    ri.pred_regs = pred_regs;
    ri.upred_regs = upred_regs;
    ri.constant = constant;

    if ((num_regs & 0xff) || unum_regs || (constant != 0)) {
        va_list vl;
        va_start(vl, unum_regs);

        for (int i = 0; i < (num_regs & 0xff); i++) {
            uint32_t val = va_arg(vl, uint32_t);

            /* collect register values from other threads */
            for (int tid = 0; tid < 32; tid++) {
              ri.reg_vals[tid][i] = __shfl_sync(active_mask, val, tid);
            }
        }

	if(first_laneid == laneid) {
	  for (int i = 0; i < unum_regs; i++) {
            uint32_t val = va_arg(vl, uint32_t);

	    ri.ureg_vals[i] = val;
	  }

	  if(pred_regs)
	    ri.pred_vals = va_arg(vl, uint32_t);

	  if(upred_regs)
	    ri.upred_vals = va_arg(vl, uint32_t);

	  if(constant == 1) {
	    ri.c.constant32 = va_arg(vl, uint32_t);
	  } else if (constant == 2) {
	    ri.c.constant64 = va_arg(vl, uint64_t);
	  }
	}

        va_end(vl);
    }

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ri, sizeof(reg_info_t));
    }
}
