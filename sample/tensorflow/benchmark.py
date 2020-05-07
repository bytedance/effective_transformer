# Copyright (C) 2019 ByteDance Inc

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
import argparse
import modeling
import numpy as np
import os
import tensorflow as tf
import effective_transformer

# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(args):
  bert_config = modeling.BertConfig.from_json_file(args.config)
  bert_config.hidden_dropout_prob = 0.0
  bert_config.attention_probs_dropout_prob = 0.0

  batch_size  = args.batch_size
  avg_seq_len = args.avg_seq_length
  max_seq_len = args.max_seq_length
  tf_dtype = tf.float16 if args.precision =='fp16' else tf.float32

  # fake input array length
  input_len = np.random.randint(
    low = 2 * avg_seq_len - max_seq_len, high = max_seq_len + 1, 
    size = (batch_size), dtype = np.int32)
  valid_word_num = sum(input_len)

  # fake input id and mask
  input_ids  = np.random.randint(
    low = 0, high = bert_config.vocab_size, 
    size = (batch_size, max_seq_len), dtype = np.int32)
  input_mask = np.zeros((batch_size, max_seq_len), dtype = np.int32)
  for b_idx, s_len in enumerate(input_len) :
    input_mask[b_idx][:s_len] = 1

  input_ids_tensor  = tf.convert_to_tensor(input_ids,  dtype = tf.int32)
  input_mask_tensor = tf.convert_to_tensor(input_mask, dtype = tf.int32)

  # fake embedding output 
  embed_output = np.random.randn(batch_size, max_seq_len, bert_config.hidden_size)
  input_tensor = tf.convert_to_tensor(embed_output, dtype = tf_dtype)

  # keep attention_mask for compatible reason
  att_mask = np.tile(input_mask, max_seq_len)
  att_mask = att_mask.reshape(batch_size, max_seq_len, max_seq_len)
  attention_mask = tf.convert_to_tensor(att_mask, dtype = tf_dtype)

  # input info
  valid_word_num = sum(input_len)
  print("Valid word num : {}/{}, avg sequence length : {:.6} ".format(
    valid_word_num, batch_size * max_seq_len, valid_word_num / batch_size))

  # bert with standard transformer
  std_bert = modeling.transformer_model(
    input_tensor                 = input_tensor,
    attention_mask               = attention_mask,
    hidden_size                  = bert_config.hidden_size,
    num_hidden_layers            = bert_config.num_hidden_layers,
    num_attention_heads          = bert_config.num_attention_heads,
    intermediate_size            = bert_config.intermediate_size,
    intermediate_act_fn          = modeling.get_activation(bert_config.hidden_act),
    hidden_dropout_prob          = bert_config.hidden_dropout_prob,
    attention_probs_dropout_prob = bert_config.attention_probs_dropout_prob,
    initializer_range            = bert_config.initializer_range,
    do_return_all_layers         = False)

  config = tf.ConfigProto()
  config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  with tf.Session(config=config) as sess:
    # init weights
    sess.run(tf.global_variables_initializer())

    # get transformer weights
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    transformer_vars = [v for v in all_vars if v.name.startswith('layer')]
    weights_value = sess.run(transformer_vars)

    # bert with effective transformer
    et_bert = effective_transformer.get_sequence_output(
      max_batch_size = batch_size,
      max_seq_length = max_seq_len,
      config         = bert_config,
      attention_mask = attention_mask,
      input_mask     = input_mask_tensor,
      from_tensor    = input_tensor,
      weights_value  = weights_value,
    )

    # diff
    val1 = sess.run(std_bert).reshape(-1, 768)
    val2 = sess.run(et_bert).reshape(-1, 768)
    diff = []
    for b_idx, s_len in enumerate(input_len) :
      for w_idx in range(s_len) :
        idx = b_idx * args.max_seq_length + w_idx
        diff.append(np.fabs(val1[idx] - val2[idx]).max())
    print("max diff : {:.6}, avg diff : {:.6}.".format(max(diff), sum(diff) / len(diff)))

    def time_inference(output_tensor) :
      iter_num = 128
      # warm up
      for i in range(10) : 
        sess.run(output_tensor)
      
      beg = datetime.now()
      for i in range(iter_num):
        sess.run(output_tensor)
      end = datetime.now()
      return (end - beg).total_seconds() * 1000 / iter_num # ms

    print("xla cost : {:.6} ms".format(time_inference(std_bert)))
    print("et  cost : {:.6} ms".format(time_inference(et_bert)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'Bert performance measuring sample.')
  parser.add_argument(
      '-c', '--config', type = str, default = 'bert_config.json', help = 'Bert config file.')
  parser.add_argument(
      '-p', '--precision', type = str, default = 'fp16', choices=['fp32', 'fp16'], help = 'Weight precision.')
  parser.add_argument(
      '-b', '--batch_size', type = int, default = 128, help = 'Batch size.')
  parser.add_argument(
      '-m', '--max_seq_length', type = int, default = 32, help = 'Max sequence length.')
  parser.add_argument(
      '-a', '--avg_seq_length', type = int, default = 20, help = 'Average sequence length.')
  args = parser.parse_args()
  main(args)
