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

#include "tf_op/bert_transformer_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include <cuda_fp16.h>
#include <mutex>
#include <map>
namespace tensorflow
{
namespace
{

void show_shape(Tensor tensor)
{
    std::vector<int> shape;
    int num_dimensions = tensor.shape().dims();
    std::cout << " ( ";
    for(int ii_dim=0; ii_dim<num_dimensions; ii_dim++) {
        std::cout << tensor.shape().dim_size(ii_dim) << " ";
    }
    std::cout << " ) " << std::endl;
}

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("EffectiveTransformer")
  .Input("from_tensor: T")
  .Input("to_tensor: T")
  .Input("attr_mask: T")
  .Input("valid_word_num: int32")
  .Input("batch_idx: int32")
  .Input("word_idx: int32")
  .Output("output: T")
  .Attr("T: {float, half}")
  .Attr("batch_size: int >= 1")
  .Attr("from_seq_len: int >= 1")
  .Attr("to_seq_len: int >= 1")
  .Attr("head_num: int >= 1")
  .Attr("size_per_head: int >= 1")
  .Attr("attr_kernel_q: tensor")
  .Attr("attr_kernel_k: tensor")
  .Attr("attr_kernel_v: tensor")
  .Attr("attr_bias_q: tensor")
  .Attr("attr_bias_k: tensor")
  .Attr("attr_bias_v: tensor")
  .Attr("attr_output_kernel: tensor")
  .Attr("attr_output_bias: tensor")
  .Attr("attr_output_layernorm_beta: tensor")
  .Attr("attr_output_layernorm_gamma: tensor")
  .Attr("inter_kernel: tensor")
  .Attr("inter_bias: tensor")
  .Attr("output_kernel: tensor")
  .Attr("output_bias: tensor")
  .Attr("output_layernorm_beta: tensor")
  .Attr("output_layernorm_gamma: tensor")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
  });
template <typename Device, typename T>
class EffectiveTransformerOp : public OpKernel
{
  public:
    const T* get_param_from_attr(OpKernelConstruction *context, std::string attr_name) {
      auto tf_tensor = weight_tensors_[attr_name];
      // debug
      // std::cout << "compute" << std::endl;
      // std::cout << attr_name << std::endl;
      // std::cout << tf_tensor.DeviceSafeDebugString() << std::endl;
      return tf_tensor.flat<T>().data();
    }

    void parse_tensor_from_attr(OpKernelConstruction *context, std::string attr_name) {
      const tensorflow::TensorProto* proto = nullptr;
      tensorflow::Tensor attr_tensor;
      // get tensor from attr first
      OP_REQUIRES_OK(context, context->GetAttr(attr_name, &proto));
      OP_REQUIRES_OK(context, context->device()->MakeTensorFromProto(
                          *proto, tensorflow::AllocatorAttributes(), &attr_tensor));
      // hold the persistent tensor
      weight_tensors_.insert({attr_name, attr_tensor});
      // debug
      // std::cout << attr_name << std::endl;
      // std::cout << attr_tensor.DeviceSafeDebugString() << std::endl;
    }

    explicit EffectiveTransformerOp(OpKernelConstruction *context) : OpKernel(context)
    {
      OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
      OP_REQUIRES_OK(context, context->GetAttr("from_seq_len", &from_seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("to_seq_len", &to_seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));

      parse_tensor_from_attr(context, "attr_kernel_q");
      parse_tensor_from_attr(context, "attr_kernel_k");
      parse_tensor_from_attr(context, "attr_kernel_v");
      parse_tensor_from_attr(context, "attr_bias_q");
      parse_tensor_from_attr(context, "attr_bias_k");
      parse_tensor_from_attr(context, "attr_bias_v");
      parse_tensor_from_attr(context, "attr_output_kernel");
      parse_tensor_from_attr(context, "attr_output_bias");
      parse_tensor_from_attr(context, "attr_output_layernorm_beta");
      parse_tensor_from_attr(context, "attr_output_layernorm_gamma");
      parse_tensor_from_attr(context, "inter_kernel");
      parse_tensor_from_attr(context, "inter_bias");
      parse_tensor_from_attr(context, "output_kernel");
      parse_tensor_from_attr(context, "output_bias");
      parse_tensor_from_attr(context, "output_layernorm_beta");
      parse_tensor_from_attr(context, "output_layernorm_gamma");

      OP_REQUIRES(context, (from_seq_len_ == to_seq_len_),
          errors::InvalidArgument("Only support from_seq_len == to_seq_len"));
      // std::cout << "parse attrs, done" << std::endl;

      param.attr_kernel_Q = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_kernel_q"));
      param.attr_kernel_K = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_kernel_k"));
      param.attr_kernel_V = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_kernel_v"));
      param.attr_bias_Q = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_bias_q"));
      param.attr_bias_K = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_bias_k"));
      param.attr_bias_V = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_bias_v"));
      param.attr_output_kernel = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_output_kernel"));
      param.attr_output_bias = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_output_bias"));
      param.attr_output_layernorm_beta = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_output_layernorm_beta"));
      param.attr_output_layernorm_gamma = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "attr_output_layernorm_gamma"));
      param.inter_kernel = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "inter_kernel"));
      param.inter_bias = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "inter_bias"));
      param.output_kernel = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "output_kernel"));
      param.output_bias = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "output_bias"));
      param.output_layernorm_beta = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "output_layernorm_beta"));
      param.output_layernorm_gamma = reinterpret_cast<const DataType_ *>(get_param_from_attr(context, "output_layernorm_gamma"));
      OP_REQUIRES(context, param.attr_kernel_Q != nullptr, errors::InvalidArgument("attr_kernel_Q is null"));
      OP_REQUIRES(context, param.attr_kernel_K != nullptr, errors::InvalidArgument("attr_kernel_K is null"));
      OP_REQUIRES(context, param.attr_kernel_V != nullptr, errors::InvalidArgument("attr_kernel_V is null"));
      OP_REQUIRES(context, param.attr_bias_Q != nullptr, errors::InvalidArgument("attr_bias_Q is null"));
      OP_REQUIRES(context, param.attr_bias_K != nullptr, errors::InvalidArgument("attr_bias_K is null"));
      OP_REQUIRES(context, param.attr_bias_V != nullptr, errors::InvalidArgument("attr_bias_V is null"));
      OP_REQUIRES(context, param.attr_output_kernel != nullptr, errors::InvalidArgument("attr_output_kernel is null"));
      OP_REQUIRES(context, param.attr_output_bias != nullptr, errors::InvalidArgument("attr_output_bias is null"));
      OP_REQUIRES(context, param.attr_output_layernorm_beta != nullptr, errors::InvalidArgument("attr_output_layernorm_beta is null"));
      OP_REQUIRES(context, param.attr_output_layernorm_gamma != nullptr, errors::InvalidArgument("attr_output_layernorm_gamma is null"));
      OP_REQUIRES(context, param.inter_kernel != nullptr, errors::InvalidArgument("inter_kernel is null"));
      OP_REQUIRES(context, param.inter_bias != nullptr, errors::InvalidArgument("inter_bias is null"));
      OP_REQUIRES(context, param.output_kernel != nullptr, errors::InvalidArgument("output_kernel is null"));
      OP_REQUIRES(context, param.output_bias != nullptr, errors::InvalidArgument("output_bias is null"));
      OP_REQUIRES(context, param.output_layernorm_beta != nullptr, errors::InvalidArgument("output_layernorm_beta is null"));
      OP_REQUIRES(context, param.output_layernorm_gamma != nullptr, errors::InvalidArgument("output_layernorm_gamma is null"));
      // std::cout << "init param from attrs, done" << std::endl;

      // init cublas handle
      try
      {
        check_cuda_error(cublasCreate(&cublas_handle_));
      }
      catch(std::runtime_error& error)
      {
        OP_REQUIRES(context, false, errors::Internal(error.what()));
      }
      // std::cout << "init cublas handle, done" << std::endl;
    }

    void Compute(OpKernelContext *context) override
    {
      OP_REQUIRES(context, context->num_inputs() == 6, errors::InvalidArgument("Less input arguments"));

      param.cublas_handle = cublas_handle_;
      param.from_tensor = reinterpret_cast<const DataType_ *>(context->input(0).flat<T>().data());
      param.to_tensor = reinterpret_cast<const DataType_ *>(context->input(1).flat<T>().data());
      param.attr_mask = reinterpret_cast<const DataType_ *>(context->input(2).flat<T>().data());
      param.valid_word_num = reinterpret_cast<const int *>(context->input(3).flat<int>().data());
      param.batch_idx      = reinterpret_cast<const int *>(context->input(4).flat<int>().data());
      param.word_idx       = reinterpret_cast<const int *>(context->input(5).flat<int>().data());

      OP_REQUIRES(context, param.from_tensor != nullptr, errors::InvalidArgument("from tensor is null"));
      OP_REQUIRES(context, param.to_tensor != nullptr, errors::InvalidArgument("to tensor is null"));
      OP_REQUIRES(context, param.valid_word_num != nullptr, errors::InvalidArgument("valid_word_num is null"));
      OP_REQUIRES(context, param.batch_idx      != nullptr, errors::InvalidArgument("batch_idx is null"));
      OP_REQUIRES(context, param.word_idx       != nullptr, errors::InvalidArgument("word_idx is null"));


      Tensor *output = nullptr;

      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, context->input(0).shape(), &output));

      param.transformer_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

      // std::cout << "bert : " << std::endl;
      // show_shape(context->input(0));

      batch_size_ = context->input(0).shape().dim_size(0);
      TransformerParam tsfp = TransformerParam(batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_);

      std::lock_guard<std::mutex> guard(cublas_mutex_);

      OP_REQUIRES_OK(
          context,
          functor::EffectiveTransformerOpFunctor<Device, T>::Compute(
            context,
            param,
            tsfp));
    }

  private:
    int batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_;
    typedef TransformerTFTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
    cublasHandle_t cublas_handle_;
    std::mutex cublas_mutex_;
    EncoderInitParam<DataType_> param; //init param here
    std::map<std::string, tensorflow::Tensor> weight_tensors_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                       \
    REGISTER_KERNEL_BUILDER(                                                                  \
        Name("EffectiveTransformer").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        EffectiveTransformerOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif

/// ************************* Transformer input parser *************************
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("EffectiveTransformerInput")
  .Input("from_tensor: T")
  .Input("mask: int32")
  .Output("to_tensor: T")
  .Output("valid_word_num: int32")
  .Output("batch_idx: int32")
  .Output("word_idx: int32")
  .Attr("T: {float, half}")
  .Attr("batch_size: int >= 1")
  .Attr("from_seq_len: int >= 1")
  .Attr("to_seq_len: int >= 1")
  .Attr("head_num: int >= 1")
  .Attr("size_per_head: int >= 1")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      int batch_size, from_seq_len, to_seq_len, head_num, size_per_head;
      c->GetAttr("batch_size", &batch_size);
      c->GetAttr("from_seq_len", &from_seq_len);
      c->GetAttr("to_seq_len", &to_seq_len);
      c->GetAttr("head_num", &head_num);
      c->GetAttr("size_per_head", &size_per_head);
      c->set_output(0, c->input(0));
      c->set_output(1, c->MakeShape({1}));
      c->set_output(2, c->input(1));
      c->set_output(3, c->input(1));
      return Status::OK();
      });

template <typename Device, typename T>
class EffectiveTransformerInputOp : public OpKernel
{
  public:
    explicit EffectiveTransformerInputOp(OpKernelConstruction *context) : OpKernel(context)
    {
      OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
      OP_REQUIRES_OK(context, context->GetAttr("from_seq_len", &from_seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("to_seq_len", &to_seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));

      OP_REQUIRES(context, (from_seq_len_ == to_seq_len_),
          errors::InvalidArgument("Only support from_seq_len == to_seq_len"));
    }

    void Compute(OpKernelContext *context) override
    {
      OP_REQUIRES(context, context->num_inputs() == 2, errors::InvalidArgument("Less input arguments"));

      EncoderInputInitParam<DataType_> param; //init param here
      param.from_tensor = reinterpret_cast<const DataType_ *>(context->input(0).flat<T>().data());
      param.mask        = reinterpret_cast<const int *>(context->input(1).flat<int>().data());

      OP_REQUIRES(context, param.from_tensor != nullptr, errors::InvalidArgument("from tensor is null"));
      OP_REQUIRES(context, param.mask != nullptr, errors::InvalidArgument("mask is null"));

      Tensor *to_tensor      = nullptr;
      Tensor *valid_word_num = nullptr;
      Tensor *batch_idx      = nullptr;
      Tensor *word_idx       = nullptr;

      // std::cout << "bert iput : " << std::endl;

      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, context->input(0).shape(), &to_tensor));
      OP_REQUIRES_OK(
          context,
          context->allocate_output(1, {1}, &valid_word_num));
      OP_REQUIRES_OK(
          context,
          context->allocate_output(2, context->input(1).shape(), &batch_idx));
      OP_REQUIRES_OK(
          context,
          context->allocate_output(3, context->input(1).shape(), &word_idx));
      // show_shape(context->input(0));
      // show_shape(context->input(1));
      // show_shape(*to_tensor);

      param.to_tensor       = reinterpret_cast<DataType_ *>(to_tensor->flat<T>().data());
      param.valid_word_num  = reinterpret_cast<int *>(valid_word_num->flat<int>().data());
      param.batch_idx       = reinterpret_cast<int *>(batch_idx->flat<int>().data());
      param.word_idx        = reinterpret_cast<int *>(word_idx->flat<int>().data());

      param.batch_size_ = context->input(0).shape().dim_size(0);
      param.from_seq_len_ = from_seq_len_;
      param.head_num_ = head_num_;
      param.size_per_head_ = size_per_head_;
      OP_REQUIRES_OK(
          context,
          functor::EffectiveTransformerInputOpFunctor<Device, T>::Compute(
            context,
            param));
    }
    private:
    int batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_;
    typedef TransformerTFTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                       \
REGISTER_KERNEL_BUILDER(Name("EffectiveTransformerInput")          \
                            .Device(DEVICE_GPU)               \
                            .TypeConstraint<T>("T"),          \
                        EffectiveTransformerInputOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU
#endif

/// ************************* Transformer output parser *************************
REGISTER_OP("EffectiveTransformerOutput")
  .Input("from_tensor: T")
  .Input("valid_word_num: int32")
  .Input("batch_idx: int32")
  .Input("word_idx: int32")
  .Output("to_tensor: T")
  .Attr("T: {float, half}")
  .Attr("batch_size: int >= 1")
  .Attr("from_seq_len: int >= 1")
  .Attr("to_seq_len: int >= 1")
  .Attr("head_num: int >= 1")
  .Attr("size_per_head: int >= 1")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      int batch_size, from_seq_len, to_seq_len, head_num, size_per_head;
      c->GetAttr("batch_size", &batch_size);
      c->GetAttr("from_seq_len", &from_seq_len);
      c->GetAttr("to_seq_len", &to_seq_len);
      c->GetAttr("head_num", &head_num);
      c->GetAttr("size_per_head", &size_per_head);
      c->set_output(0, c->input(0));
      return Status::OK();
      });

template <typename Device, typename T>
class EffectiveTransformerOutputOp : public OpKernel
{
  public:
    explicit EffectiveTransformerOutputOp(OpKernelConstruction *context) : OpKernel(context)
    {
      OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
      OP_REQUIRES_OK(context, context->GetAttr("from_seq_len", &from_seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("to_seq_len", &to_seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));

      OP_REQUIRES(context, (from_seq_len_ == to_seq_len_),
          errors::InvalidArgument("Only support from_seq_len == to_seq_len"));
    }

    void Compute(OpKernelContext *context) override
    {
      /// 1. get input addr and check
      OP_REQUIRES(context, context->num_inputs() == 4, errors::InvalidArgument("Less input arguments"));

      EncoderOutputInitParam<DataType_> param; //init param here
      param.from_tensor    = reinterpret_cast<const DataType_ *>(context->input(0).flat<T>().data());
      param.valid_word_num = reinterpret_cast<const int *>(context->input(1).flat<int>().data());
      param.batch_idx      = reinterpret_cast<const int *>(context->input(2).flat<int>().data());
      param.word_idx       = reinterpret_cast<const int *>(context->input(3).flat<int>().data());

      OP_REQUIRES(context, param.from_tensor != nullptr,    errors::InvalidArgument("from tensor is null"));
      OP_REQUIRES(context, param.valid_word_num != nullptr, errors::InvalidArgument("valid_word_num is null"));
      OP_REQUIRES(context, param.batch_idx != nullptr,      errors::InvalidArgument("batch_idx is null"));
      OP_REQUIRES(context, param.word_idx != nullptr,       errors::InvalidArgument("word_idx is null"));

      /// 2. allocate output
      Tensor *to_tensor      = nullptr;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, context->input(0).shape(), &to_tensor));
      param.to_tensor = reinterpret_cast<DataType_ *>(to_tensor->flat<T>().data());

      /// 3. launch kernel
      param.batch_size_ = context->input(0).shape().dim_size(0);;
      param.from_seq_len_ = from_seq_len_;
      param.head_num_ = head_num_;
      param.size_per_head_ = size_per_head_;
      OP_REQUIRES_OK(
          context,
          functor::EffectiveTransformerOutputOpFunctor<Device, T>::Compute(
            context,
            param));
    }
    private:
    int batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_;
    typedef TransformerTFTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                        \
REGISTER_KERNEL_BUILDER(Name("EffectiveTransformerOutput")          \
                            .Device(DEVICE_GPU)                \
                            .TypeConstraint<T>("T"),           \
                        EffectiveTransformerOutputOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU
#endif


} //namespace
} //namespace tensorflow
