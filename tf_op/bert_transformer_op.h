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

#ifndef TENSORFLOW_CORE_KERNELS_MULTIHEADATTR_OP_H_
#define TENSORFLOW_CORE_KERNELS_MULTIHEADATTR_OP_H_

#include "common.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cublas_v2.h>
using namespace effectivetransformer;
namespace tensorflow
{
  template <typename T> class TransformerTFTraits;

  template <>
  class TransformerTFTraits<float>
  {
    public:
      typedef float DataType;
      static const OperationType OpType = OperationType::FP32;
  };

  template <>
  class TransformerTFTraits<Eigen::half>
  {
    public:
      typedef __half DataType;
      static const OperationType OpType = OperationType::HALF;
  };

  class TransformerParam {
    public:
      int batch_size_;
      int from_seq_len_;
      int to_seq_len_;   /// now only support from_seq_len_ == to_seq_len_
      int head_num_;
      int size_per_head_;

      TransformerParam(int batch_size, int from_seq_len,
                       int to_seq_len,  int head_num, int size_per_head)
        : batch_size_(batch_size), from_seq_len_(from_seq_len),
          to_seq_len_(to_seq_len), head_num_(head_num), size_per_head_(size_per_head)
      {
      }
  };

  template<typename T>
  class EncoderInitParam
  {
    public:
      const T* from_tensor;
      const T* to_tensor;
      const T* attr_kernel_Q;
      const T* attr_kernel_K;
      const T* attr_kernel_V;
      const T* attr_bias_Q;
      const T* attr_bias_K;
      const T* attr_bias_V;
      const T* attr_mask;
      const T* attr_output_kernel;
      const T* attr_output_bias;
      const T* attr_output_layernorm_gamma;
      const T* attr_output_layernorm_beta;
      const T* inter_kernel;
      const T* inter_bias;
      const T* output_kernel;
      const T* output_bias;
      const T* output_layernorm_gamma;
      const T* output_layernorm_beta;
      const int* batch_idx;
      const int* word_idx;
      const int* valid_word_num;
      T* transformer_out;
      cublasHandle_t cublas_handle;
      cudaStream_t stream;
  };

  template<typename T>
  class EncoderInputInitParam
  {
    public:
      const T* from_tensor;
      const int* mask;
      int* valid_word_num;
      int* batch_idx;
      int* word_idx;
      int batch_size_;
      int from_seq_len_;
      int head_num_;
      int size_per_head_;
      T* to_tensor;
      cudaStream_t stream;
  };

  template<typename T>
  class EncoderOutputInitParam
  {
    public:
      const T* from_tensor;
      const int* valid_word_num;
      const int* batch_idx;
      const int* word_idx;
      int batch_size_;
      int from_seq_len_;
      int head_num_;
      int size_per_head_;
      T* to_tensor;
      cudaStream_t stream;
  };


  namespace functor
  {
    /// ***************************** Transformer op ******************************
    template <typename Device, typename T>
    struct EffectiveTransformerOpFunctor
    {
      typedef typename TransformerTFTraits<T>::DataType DataType_;
      static Status Compute(OpKernelContext *context,
        EncoderInitParam<DataType_ > param,
        TransformerParam t_param);
    };

    /// ************************* Transformer input parser *************************
    template <typename Device, typename T>
    struct EffectiveTransformerInputOpFunctor
    {
      typedef typename TransformerTFTraits<T>::DataType DataType_;
      static Status Compute(OpKernelContext *context,
                            EncoderInputInitParam<DataType_ > param);
    };

    /// ************************* Transformer output parser *************************
    template <typename Device, typename T>
    struct EffectiveTransformerOutputOpFunctor
    {
      typedef typename TransformerTFTraits<T>::DataType DataType_;
      static Status Compute(OpKernelContext *context,
                            EncoderOutputInitParam<DataType_ > param);
    };

  } //namespace functor
} //namespace tensorflow
#endif
