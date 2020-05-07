/*
 * Copyright (C) 2020 ByteDance Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace effectivetransformer{
namespace cuda{

template<typename T>
void add_QKV_bias_padding_kernelLauncher(
    T* Q, const T* bias_Q, 
    T* K, const T* bias_K, 
    T* V, const T* bias_V, 
    T* q_buf_, T* k_buf_, T* v_buf_,  
    const int valid_word_num,
    const int batch_size, const int seq_len, 
    const int head_num, const int size_per_head, 
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream);

template <typename T> 
void softmax_kernel_kernelLauncher(
    T* qk_buf_, const T* attr_mask, 
    const int batch_size, const int head_num, const int seq_len, 
    const T scaler,
    const cudaStream_t stream);

template <typename T> 
void transpose_rm_padding_kernelLauncher(
    T* src, T* dst,
    const int valid_word_num,
    const int batch_size, const int seq_len, 
    const int head_num, const int size_per_head,
    const int* batch_idx, const int* word_idx,
    const cudaStream_t stream);
}//namespace cuda
}//namespace effectivetransformer
