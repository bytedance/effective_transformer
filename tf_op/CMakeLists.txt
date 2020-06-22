# bytedance 

cmake_minimum_required(VERSION 3.8)

set(tf_bert_transformer_files
  bert_transformer_op.cc
  bert_transformer_op.cu.cc
  ../cuda/attention.cu
  ../cuda/cuda_kernels.cu
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DGOOGLE_CUDA=1)
add_definitions(-DNDEBUG)

add_library(tf_effectivetransformer SHARED ${tf_bert_transformer_files})
set_target_properties(tf_effectivetransformer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(tf_effectivetransformer PRIVATE -lcublas -lcudart -l:libtensorflow_framework.so.2 -l:_pywrap_tensorflow_internal.so ${CMAKE_THREAD_LIBS_INIT})
