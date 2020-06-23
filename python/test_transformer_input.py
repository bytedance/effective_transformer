import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
import os


__lib_tf = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "libtf_effectivetransformer.so")


def tf_load_op_library(path):
    lib = tf.load_op_library(path)
    return lib

transformer_op_module = tf_load_op_library(__lib_tf)


batch_size = 3
max_seq_len = 5
head_num = 1
size_per_head = 1
input_embedding = np.random.randn(batch_size, max_seq_len, head_num * size_per_head)
from_tensor = tf.convert_to_tensor(input_embedding, tf.float32)
input_mask = np.zeros((batch_size, max_seq_len), dtype = np.int32)
for b_idx, s_len in enumerate([2,3,4]):
  input_mask[b_idx][:s_len] = 1
input_mask = tf.convert_to_tensor(input_mask, dtype = tf.int32)


@tf.function(experimental_compile=True)
def func(from_tensor, input_mask):
  comp_from_tensor, valid_word_num_tensor, batch_idx_tensor, word_idx_tensor = \
    transformer_op_module.bert_transformer_input(from_tensor, input_mask,
                                                 batch_size=batch_size, from_seq_len=max_seq_len, to_seq_len=max_seq_len,
                                                 head_num=head_num, size_per_head=size_per_head)
  return comp_from_tensor

print(from_tensor)
print(func(from_tensor, input_mask))
