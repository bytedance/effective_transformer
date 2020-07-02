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

#define EIGEN_USE_GPU


#include "cuda/scatter_nd.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


#include <cuda_fp16.h>
#include <map>
#include <mutex>



namespace tensorflow {
namespace {

enum class OperationType {FP32, HALF};

template <typename T> class TFTraits;

template <>
class TFTraits<float>
{
  public:
    typedef float DataType;
    static const OperationType OpType = OperationType::FP32;
};

template <>
class TFTraits<Eigen::half>
{
  public:
    typedef __half DataType;
    static const OperationType OpType = OperationType::HALF;
};


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("ScatterNdMemcpy")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Input("shape: int32")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("Tindices: {int32}");

template <typename Device, typename T, typename Index>
class ScatterNdMemcpyOp : public OpKernel {
public:
  const T *get_param_from_attr(OpKernelContext *context,
                               std::string attr_name) {}

  void parse_tensor_from_attr(OpKernelConstruction *context,
                              std::string attr_name) {
    const tensorflow::TensorProto *proto = nullptr;
    tensorflow::Tensor attr_tensor;
    // get tensor from attr first
    OP_REQUIRES_OK(context, context->GetAttr(attr_name, &proto));
    OP_REQUIRES_OK(
        context, context->device()->MakeTensorFromProto(
                     *proto, tensorflow::AllocatorAttributes(), &attr_tensor));
  }

  explicit ScatterNdMemcpyOp(OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {

    const Tensor &indices_tensor = context->input(0);
    const Tensor &updates_tensor = context->input(1);
    const Tensor &shape_tensor = context->input(2);

    const int src_m = updates_tensor.shape().dim_size(0);
    const int src_n = updates_tensor.shape().dim_size(1);

    const DataType_ *src_ptr =
        reinterpret_cast<const DataType_ *>(updates_tensor.flat<T>().data());
    const int *indices_ptr =
        reinterpret_cast<const int *>(indices_tensor.flat<Index>().data());
    
    const cudaStream_t& stream = context->eigen_device<GPUDevice>().stream();

    auto vec = shape_tensor.flat<Index>();

    int dst_m = vec.data()[0];
    int dst_n = vec.data()[1];
    // printf("src [%d, %d] dst [%d, %d]\n", src_m, src_n, dst_m, dst_n);
    // get cuda stream
    // allocate output
    Tensor *output = nullptr;
    TensorShape out_shape({dst_m, dst_n});

    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    DataType_ *dst_ptr =
        reinterpret_cast<DataType_ *>(output->flat<T>().data());

    cudaMemset(dst_ptr, -100000.0, sizeof(DataType_) * (dst_m * dst_n));
    scatterNd_kernel_launcher(src_ptr, indices_ptr, dst_ptr, src_m, dst_m,
                              src_n, stream);

    // cudaMemcpy(dst_ptr, src_ptr, sizeof(DataType_) * (src_m * src_n),
    // cudaMemcpyDeviceToDevice);
  }

private:
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#if GOOGLE_CUDA

#define REGISTER_GPU(T, index_type)                                            \
  REGISTER_KERNEL_BUILDER(Name("ScatterNdMemcpy")                              \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .TypeConstraint<index_type>("Tindices")          \
                              .HostMemory("shape"),                            \
                          ScatterNdMemcpyOp<GPUDevice, T, index_type>)
REGISTER_GPU(Eigen::half, int32);
REGISTER_GPU(float, int32);
#undef REGISTER_GPU

#endif

} // namespace
} // namespace tensorflow