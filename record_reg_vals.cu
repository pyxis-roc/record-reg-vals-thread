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

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <map>
#include <vector>
#include <unordered_set>
#include <cstring>
#include <errno.h>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#define USE_ASYNC_STREAM
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
#include "common.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;
__managed__ warp_selection_t wst;

/* receiving thread and its control variables */
pthread_t recv_thread;

enum class RecvThreadState {
    WORKING,
    STOP,
    FINISHED,
};
volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

/* lock */
pthread_mutex_t cuda_event_mutex;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t instr_ipoint_pre = 0;

int verbose = 0;
std::string output_filename;
std::string warp_selection;
std::string kernel_name;
/* opcode to id map and reverse map  */
std::map<std::string, int> sass_to_id_map;
std::map<int, std::string> id_to_sass_map;

uint32_t kernel_begin = 0;
uint32_t kernel_end = UINT32_MAX;

warp_selection_t wst_host;

FILE *output_file = stdout;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(kernel_begin, "KERNEL_BEGIN", 0, "Instrument starting from this kernel");
    GET_VAR_INT(kernel_end, "KERNEL_END", UINT32_MAX, "Instrument up to (but excluding) this kernel launch");
    GET_VAR_STR(kernel_name, "KERNEL_NAME", "Instrument only this kernel, must be a mangled name");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_STR(output_filename, "TRACE_FILE", "Output trace to file");
    GET_VAR_INT(instr_ipoint_pre, "INSTR_IPOINT_PRE", 0, "Instrument before instruction");
    GET_VAR_STR(warp_selection, "INSTR_WARP", "Select warp as ctax,ctay,ctaz,warp_id")

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    if (output_filename.size() > 0) {
      output_file = fopen(output_filename.c_str(), "w");
      if (output_file == NULL) {
        fprintf(stderr, "ERROR: Unable to open %s for writing (%s) \n",
                output_filename.c_str(), strerror(errno));
      } else {
	fprintf(stderr, "Writing trace to %s\n", output_filename.c_str());
      }
    }

    if (warp_selection.size() == 0) {
      wst_host.cta_x = 0;
      wst_host.cta_y = 0;
      wst_host.cta_z = 0;
      wst_host.warp_id = 0;
    }
    else {
      char *source = strdup(warp_selection.c_str());
      if(!source) {
	fprintf(stderr, "ERROR: Unable to allocate memory, ignoring warp selection.\n");
	wst_host.cta_x = 0;
	wst_host.cta_y = 0;
	wst_host.cta_z = 0;
	wst_host.warp_id = 0;
      } else {
	const char *parts[5];
	char *saveptr;
	int partcount = 0;

	parts[0] = strtok_r(source, ",", &saveptr);
	while(partcount < 5 && (parts[partcount++] != NULL)) {
	  parts[partcount] = strtok_r(NULL, ",", &saveptr);
	}
	if(partcount == 5 && parts[4] == NULL) {
	  wst_host.cta_x = atoi(parts[0]);
	  wst_host.cta_y = atoi(parts[1]);
	  wst_host.cta_z = atoi(parts[2]);
	  wst_host.warp_id = atoi(parts[3]);
	}
	free(source);
      }
    }

    fprintf(stderr, "Instrumenting warp: %d,%d,%d - %d\n",
	    wst_host.cta_x, wst_host.cta_y, wst_host.cta_z, wst_host.warp_id);
}
/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;
static uint32_t kernel_count = 0;
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {

    kernel_count++;

    if(kernel_count < kernel_begin || kernel_count >= kernel_end) {
      return;
    }

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    if(!kernel_name.empty()) {
      const std::string fname{nvbit_get_func_name(ctx, func, true)};
      if(fname.compare(kernel_name) != 0)
	return;
    }

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        if (verbose) {
            printf("Kernel #%d: Inspecting function %s at address 0x%lx\n",
                   kernel_count, nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
        }

        uint32_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
                cnt++;
                continue;
            }
            if (verbose) {
                instr->printDecoded();
            }

            if (sass_to_id_map.find(instr->getSass()) ==
                sass_to_id_map.end()) {
                int opcode_id = sass_to_id_map.size();
                sass_to_id_map[instr->getSass()] = opcode_id;
                id_to_sass_map[opcode_id] = std::string(instr->getSass());
            }

            int opcode_id = sass_to_id_map[instr->getSass()];
            std::vector<int> reg_num_list;
            std::vector<int> ureg_num_list;
	    uint32_t pred_list = 0;
	    uint32_t upred_list = 0;

	    int32_t constant = 0;
	    int32_t bankid, bankoffset;
	    int width =  instr->getSize() / 4;
	    int owidth = 0;
	    if (width < 1) width = 1;
	    if(strncmp("IMAD.WIDE", instr->getOpcode(), 9) == 0)
	      width = 2;
	    else if(strncmp("HMMA.16816.F32", instr->getOpcode(), 14) == 0)
	      width = 4;
	    else if(strncmp("LDG.E.LTC128B.CONSTANT", instr->getOpcode(), 22) == 0)
	      width = 4;

	    owidth = width;

            /* iterate on the operands */
            for (int i = 0; i < instr->getNumOperands(); i++) {
                /* get the operand "i" */
                const InstrType::operand_t *op = instr->getOperand(i);
                if (op->type == InstrType::OperandType::REG) {
                    for (int reg_idx = 0; reg_idx < width; reg_idx++) {
                        reg_num_list.push_back(op->u.reg.num + reg_idx);
                    }
                } else if (op->type == InstrType::OperandType::UREG) {
		  for (int reg_idx = 0; reg_idx < width; reg_idx++) {
                        ureg_num_list.push_back(op->u.reg.num + reg_idx);
                    }
		} else if (op->type == InstrType::OperandType::CBANK) {
		  if(op->u.cbank.has_imm_offset) {
		    constant = instr->getSize() / 4; // use this instead of width
		    if(instr->getSize() < 4)
		      constant = 1; // LDC.U8
		    if(constant > 2) constant = 2;
		    bankid = op->u.cbank.id;
		    bankoffset = op->u.cbank.imm_offset;
		  }
		} else if (op->type == InstrType::OperandType::PRED) {
                  if(op->u.pred.num != InstrType::PT)
                    pred_list |= 1 << op->u.pred.num; // no implicit pred registers?
		} else if(op->type == InstrType::OperandType::UPRED) {
                  if(op->u.pred.num != InstrType::UPT)
                    upred_list |= 1 << op->u.pred.num; // no implicit pred registers?
		}
		width = 1;
            }
            /* insert call to the instrumentation function with its
             * arguments */
            if(instr_ipoint_pre)
	      nvbit_insert_call(instr, "record_reg_val_thread", IPOINT_BEFORE);
	    else
	      nvbit_insert_call(instr, "record_reg_val_thread", IPOINT_AFTER);
            /* guard predicate value */
            nvbit_add_call_arg_guard_pred_val(instr);
            /* opcode id */
            nvbit_add_call_arg_const_val32(instr, opcode_id);
            /* idx */
            nvbit_add_call_arg_const_val32(instr, instr->getIdx());
            /* add pointer to channel_dev*/
            nvbit_add_call_arg_const_val64(instr,
                                           (uint64_t)&channel_dev);
            /* add pointer to selection*/
            nvbit_add_call_arg_const_val64(instr,
                                           (uint64_t)&wst);

	    nvbit_add_call_arg_const_val32(instr, constant);
            /* how many register values are passed next */
            nvbit_add_call_arg_const_val32(instr, pred_list);
	    nvbit_add_call_arg_const_val32(instr, upred_list);
            nvbit_add_call_arg_const_val32(instr, (owidth << 8) | reg_num_list.size());
            nvbit_add_call_arg_const_val32(instr, ureg_num_list.size());

            for (int num : reg_num_list) {
                /* last parameter tells it is a variadic parameter passed to
                 * the instrument function record_reg_val() */
                nvbit_add_call_arg_reg_val(instr, num, true);
            }
            for (int num : ureg_num_list) {
                /* last parameter tells it is a variadic parameter passed to
                 * the instrument function record_reg_val() */
                nvbit_add_call_arg_ureg_val(instr, num, true);
            }

	    if(pred_list)
	      nvbit_add_call_arg_pred_reg(instr, true);

	    if(upred_list)
	      nvbit_add_call_arg_upred_reg(instr, true);

	    if(constant) {
	      if(constant == 1) {
		nvbit_add_call_arg_cbank_val(instr, bankid, bankoffset, true);
	      } else if(constant == 2) {
		nvbit_add_call_arg_cbank_val(instr, bankid, bankoffset, true);
		nvbit_add_call_arg_cbank_val(instr, bankid, bankoffset+4, true);
	      }
	    }
            cnt++;
        }
    }
}

__global__ void flush_channel() {
    /* push memory access with negative cta id to communicate the kernel is
     * completed */
    reg_info_t ri;
    ri.cta_id_x = -1;
    channel_dev.push(&ri, sizeof(reg_info_t));

    /* flush channel */
    channel_dev.flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    pthread_mutex_lock(&cuda_event_mutex);

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;

    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            /* Make sure GPU is idle */
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            int nregs = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

            int shmem_static_nbytes = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                   func));

            instrument_function_if_needed(ctx, func);

            nvbit_enable_instrumented(ctx, func, true);

            if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
              cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
	      if(output_file)
                fprintf(output_file,
                    "Kernel %d:%s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
                    "%d - shmem %d - cuda stream id %ld - ipoint_pre %d\n", kernel_count,
                    nvbit_get_func_name(ctx, func),
                    p->config->gridDimX, p->config->gridDimY,
                    p->config->gridDimZ, p->config->blockDimX,
                    p->config->blockDimY, p->config->blockDimZ, nregs,
                    shmem_static_nbytes + p->config->sharedMemBytes,
			(uint64_t)p->config->hStream,
			instr_ipoint_pre);
            } else {
              cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
              if (output_file)
		fprintf(output_file,
                    "Kernel %d:%s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
		     "%d - shmem %d - cuda stream id %ld - ipoint_pre %d\n", kernel_count,
                    nvbit_get_func_name(ctx, func), p->gridDimX, p->gridDimY,
                    p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ, nregs,
			shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream,
			instr_ipoint_pre);
            }
        } else {
            /* make sure current kernel is completed */
            cudaDeviceSynchronize();
            cudaError_t kernelError = cudaGetLastError();
            if (kernelError != cudaSuccess) {
                printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
                assert(0);
            }

            /* issue flush of channel so we are sure all the memory accesses
             * have been pushed */
            flush_channel<<<1, 1>>>();
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);
        }
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void *recv_thread_fun(void *) {
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);
    char *output = (char *)malloc(1024);

    while (recv_thread_done == RecvThreadState::WORKING) {
        uint32_t num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE);

        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                reg_info_t *ri =
                    (reg_info_t *)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                 */
                if (ri->cta_id_x == -1) {
                    break;
                }

		if(output_file)
                fprintf(output_file, "CTA %d,%d,%d - warp %d - %d - %s:\n", ri->cta_id_x,
                       ri->cta_id_y, ri->cta_id_z, ri->warp_id, ri->opcode_idx,
                       id_to_sass_map[ri->opcode_id].c_str());

		fprintf(output_file, "* Width: %0d\n", (ri->num_regs >> 8) & 0xff);
                for (int reg_idx = 0; reg_idx < (ri->num_regs & 0xff); reg_idx++) {
                  char *start = output;
                  size_t sz = 1024;
		  int wr = 0;
                    for (int i = 0; i < 32; i++) {
                      wr = snprintf(start, sz, "Reg%d_T%d: 0x%08x ", reg_idx, i,
                                    ri->reg_vals[i][reg_idx]);
                      if (wr >= sz)
                        break; // TRUNCATED
                      sz -= wr;
		      start += wr;
                    }
                    if (output_file)
                      fprintf(output_file, "* %s\n", output);
                }

                if(ri->pred_regs & (1 << InstrType::PR)) {
                  if((ri->num_regs & 0xff) < 7) {
                    uint32_t reg_idx = (ri->num_regs & 0xff) + 1;

                    char *start = output;
                    size_t sz = 1024;
                    int wr = 0;
                    for (int i = 0; i < 32; i++) {
                      wr = snprintf(start, sz, "PR_T%d: 0x%08x ", i,
                                    ri->reg_vals[i][reg_idx]);
                      if (wr >= sz)
                        break; // TRUNCATED
                      sz -= wr;
		      start += wr;
                    }

                    if (output_file)
                      fprintf(output_file, "* %s\n", output);
                  }
                }

                for (int reg_idx = 0; reg_idx < ri->unum_regs; reg_idx++) {
                  if (output_file)
                    fprintf(output_file,
                           "* UReg%d: 0x%08x\n", reg_idx, ri->ureg_vals[reg_idx]);
		}

		if(ri->pred_regs & ~(1 << InstrType::PR))
                  if (output_file) {
                    uint32_t regs = ri->pred_regs & ~(1 << InstrType::PR);
                    int count = __builtin_popcount(regs);
                    for(int i = 0; i < count; i++) {
                      uint32_t regndx = __builtin_ffs(regs);
                      fprintf(output_file,
                              "* Pred%d: 0x%08x\n", regndx - 1, ri->pred_vals[i]);
                      regs = regs & ~(1 << (regndx - 1));
                    }
                  }

		if(ri->upred_regs)
                  if (output_file)
                    fprintf(output_file,
			    "* UPred: 0x%08x 0x%08x\n", ri->upred_regs, ri->upred_vals);

		if(ri->constant) {
		  if(ri->constant == 1) {
                    if (output_file)
                      fprintf(output_file, "* C32: 0x%08x\n", ri->c.constant32);
		  } else if (ri->constant == 2) {
                    if (output_file)
                      fprintf(output_file, "* C64: 0x%016lx\n", ri->c.constant64);
		  }
		}
                if(output_file) fprintf(output_file, "\n");
                num_processed_bytes += sizeof(reg_info_t);
            }
        }
    }
    free(recv_buffer);
    free(output);
    recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_tool_init(CUcontext ctx) {
    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&cuda_event_mutex, &attr);

    recv_thread_done = RecvThreadState::WORKING;
    channel_host.init(0, CHANNEL_SIZE, &channel_dev, recv_thread_fun, NULL);
    nvbit_set_tool_pthread(channel_host.get_thread());

    wst.cta_x = wst_host.cta_x;
    wst.cta_y = wst_host.cta_y;
    wst.cta_z = wst_host.cta_z;
    wst.warp_id = wst_host.warp_id;
}

void nvbit_at_ctx_term(CUcontext ctx) {
    skip_callback_flag = true;
    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    recv_thread_done = RecvThreadState::STOP;
    while (recv_thread_done != RecvThreadState::FINISHED)
        ;
    channel_host.destroy(false);
    skip_callback_flag = false;
}
