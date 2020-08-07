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

import os
import tensorflow as tf

__lib_tf_1_15 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "libtf_effectivetransformer.so.1.15")


def tf_load_op_library(path):
    if not tf.__version__.startswith('1.15'):
        warnings.warn('Only tensorflow 1.15.x supported')
    lib = tf.load_op_library(path)
    return lib


transformer_op_module = tf_load_op_library(__lib_tf_1_15)


def get_sequence_output(
        config, max_batch_size, max_seq_length,
        attention_mask, input_mask, from_tensor, weights_value):
    batch_size = max_batch_size
    seq_len = max_seq_length
    head_num = config.num_attention_heads
    size_per_head = int(config.hidden_size / config.num_attention_heads)
    num_layers = config.num_hidden_layers

    # 1. rm padding
    comp_from_tensor, valid_word_num_tensor, batch_idx_tensor, word_idx_tensor = \
        transformer_op_module.effective_transformer_input(from_tensor, input_mask,
                                                     batch_size=batch_size, from_seq_len=seq_len, to_seq_len=seq_len,
                                                     head_num=head_num, size_per_head=size_per_head)

    # 2. transformer loop
    for layer_idx in range(num_layers):
        offset = layer_idx * 16
        comp_from_tensor = transformer_op_module.effective_transformer(
            comp_from_tensor,
            comp_from_tensor,
            attention_mask,
            valid_word_num_tensor, batch_idx_tensor, word_idx_tensor,
            attr_kernel_q=tf.make_tensor_proto(weights_value[offset + 0]),
            attr_kernel_k=tf.make_tensor_proto(weights_value[offset + 2]),
            attr_kernel_v=tf.make_tensor_proto(weights_value[offset + 4]),
            attr_bias_q=tf.make_tensor_proto(weights_value[offset + 1]),
            attr_bias_k=tf.make_tensor_proto(weights_value[offset + 3]),
            attr_bias_v=tf.make_tensor_proto(weights_value[offset + 5]),
            attr_output_kernel=tf.make_tensor_proto(weights_value[offset + 6]),
            attr_output_bias=tf.make_tensor_proto(weights_value[offset + 7]),
            attr_output_layernorm_beta=tf.make_tensor_proto(
                weights_value[offset + 8]),
            attr_output_layernorm_gamma=tf.make_tensor_proto(
                weights_value[offset + 9]),
            inter_kernel=tf.make_tensor_proto(weights_value[offset + 10]),
            inter_bias=tf.make_tensor_proto(weights_value[offset + 11]),
            output_kernel=tf.make_tensor_proto(weights_value[offset + 12]),
            output_bias=tf.make_tensor_proto(weights_value[offset + 13]),
            output_layernorm_beta=tf.make_tensor_proto(
                weights_value[offset + 14]),
            output_layernorm_gamma=tf.make_tensor_proto(
                weights_value[offset + 15]),
            batch_size=batch_size, from_seq_len=seq_len, to_seq_len=seq_len,
            head_num=head_num, size_per_head=size_per_head)

    # . 3. pad back
    transformer_output = transformer_op_module.effective_transformer_output(
        comp_from_tensor, valid_word_num_tensor, batch_idx_tensor, word_idx_tensor,
        batch_size=batch_size, from_seq_len=seq_len, to_seq_len=seq_len,
        head_num=head_num, size_per_head=size_per_head)

    return transformer_output
