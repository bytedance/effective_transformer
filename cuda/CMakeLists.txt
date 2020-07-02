# Bytedance

cmake_minimum_required(VERSION 3.8)

set(cuda_kernel_files
  cuda_kernels.cu
  attention.cu
  scatter_nd.cu
)

add_library(effectivetransformer STATIC ${cuda_kernel_files})
target_link_libraries(effectivetransformer PUBLIC -lcublas -lcudart ${CMAKE_THREAD_LIBS_INIT})

