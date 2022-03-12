# Copyright (C) 2019 ByteDance Inc

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(EfficientTransformer LANGUAGES CXX CUDA)
find_package(CUDA 10.0 REQUIRED)

if(NOT CUDA_PATH)
  set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
endif()
message(STATUS "CUDA_PATH    : " ${CUDA_TOOLKIT_ROOT_DIR})

if(NOT TF_PATH)
  message(FATAL_ERROR "TF_PATH must be set.")
endif()
message(STATUS "TF_PATH      : " ${TF_PATH})

list(APPEND COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
  ${TF_PATH}/
)
foreach(lib ${COMMON_LIB_DIRS})
  message(STATUS "lib path     : " ${lib})
endforeach()

list(APPEND COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
  ${TF_PATH}/include
)
foreach(inc ${COMMON_HEADER_DIRS})
  message(STATUS "include path : " ${inc})
endforeach()

list(APPEND GPU_ARCHS
  61
  70
  75
)
foreach(arch ${GPU_ARCHS})
  set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
message(STATUS "gencode      : " ${GENCODES})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -DWMMA -O3")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DWMMA -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA -O3 ${GENCODES} --std=c++11 -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr ")

message(STATUS "C_FLAGS      : " ${CMAKE_C_FLAGS})
message(STATUS "CXX_FLAGS    : " ${CMAKE_CXX_FLAGS})
message(STATUS "CUDA_FLAGS   : " ${CMAKE_CUDA_FLAGS})

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(cuda)
add_subdirectory(tf_op)

add_custom_target(copy ALL COMMENT "Copying tensorflow sample")
add_custom_command(TARGET copy
    POST_BUILD
    COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow/bert_config.json ${PROJECT_SOURCE_DIR}/build/
    COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow/benchmark.py ${PROJECT_SOURCE_DIR}/build/
    COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow/modeling.py ${PROJECT_SOURCE_DIR}/build/
)
