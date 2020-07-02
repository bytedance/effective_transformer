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
import modeling_rm_pad as modeling
import numpy as np
import os
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

def main(args):
  bert_config = modeling.BertConfig.from_json_file(args.config)
  bert_config.hidden_dropout_prob = 0.0
  bert_config.attention_probs_dropout_prob = 0.0

  batch_size  = args.batch_size
  avg_seq_len = args.avg_seq_length
  max_seq_len = args.max_seq_length
  real_max_seq_len = args.real_max_seq_length
  tf_dtype = tf.float16 if args.precision =='fp16' else tf.float32

  # fake input array length
  input_len = np.random.randint(
    low = 2 * avg_seq_len - real_max_seq_len, high = real_max_seq_len + 1, 
    size = (batch_size), dtype = np.int32)
  valid_word_num = sum(input_len)

  # fake input id and mask
  index = []
  for b_idx, s_len in enumerate(input_len) :
    tmp = [b_idx * max_seq_len + x for x in range(s_len)]
    index.extend(tmp)
  index = np.array(index).astype(np.int32)
  index_tensor = tf.placeholder(tf.int32, shape=(None, ))

  # fake embedding output 
  embed_output = np.random.randn(batch_size, max_seq_len, bert_config.hidden_size).astype(np.float16)
  input_tensor = tf.placeholder(tf_dtype, shape=(batch_size, max_seq_len, bert_config.hidden_size))

  # input info
  valid_word_num = sum(input_len)
  print("Valid word num : {}/{}, avg sequence length : {:.6} ".format(
    valid_word_num, batch_size * max_seq_len, valid_word_num / batch_size))

  # bert with standard transformer
  std_bert = modeling.transformer_model(
    input_tensor                 = input_tensor,
    index_tensor                 = index_tensor,
    hidden_size                  = bert_config.hidden_size,
    num_hidden_layers            = bert_config.num_hidden_layers,
    num_attention_heads          = bert_config.num_attention_heads,
    intermediate_size            = bert_config.intermediate_size,
    intermediate_act_fn          = modeling.get_activation(bert_config.hidden_act),
    hidden_dropout_prob          = bert_config.hidden_dropout_prob,
    attention_probs_dropout_prob = bert_config.attention_probs_dropout_prob,
    initializer_range            = bert_config.initializer_range,
    do_return_all_layers         = False,
    tf_dtype                     = tf_dtype)

  config = tf.ConfigProto()
  # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  with tf.Session(config=config) as sess:
    # init weights
    sess.run(tf.global_variables_initializer())




    
    def time_inference(output_tensor) :
      iter_num = 128

      ########
      input_len = np.random.randint(
        low = 2 * avg_seq_len - real_max_seq_len, high = real_max_seq_len + 1, 
        size = (batch_size), dtype = np.int32)
      index = []
      for b_idx, s_len in enumerate(input_len) :
        tmp = [b_idx * max_seq_len + x for x in range(s_len)]
        index.extend(tmp)
      index = np.array(index).astype(np.int32)
      #########

      # warm up
      for i in range(10):
        sess.run(output_tensor, feed_dict={input_tensor: embed_output, index_tensor: index})

      
      acc_time = None 
      
      for i in range(iter_num):
        
        ########
        input_len = np.random.randint(
          low = 2 * avg_seq_len - real_max_seq_len, high = real_max_seq_len + 1, 
          size = (batch_size), dtype = np.int32)
        index = []
        for b_idx, s_len in enumerate(input_len) :
          tmp = [b_idx * max_seq_len + x for x in range(s_len)]
          index.extend(tmp)
        index = np.array(index).astype(np.int32)
        #########
        beg = datetime.now()
        sess.run(output_tensor, feed_dict={input_tensor: embed_output, index_tensor: index})
        end = datetime.now()
        if acc_time is None:
          acc_time = end - beg
        else:
          acc_time += (end - beg)
      return acc_time.total_seconds() * 1000 / iter_num # ms

    print("xla cost : {:.6} ms".format(time_inference(std_bert)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'Bert performance measuring sample.')
  parser.add_argument(
      '-c', '--config', type = str, default = 'bert_config.json', help = 'Bert config file.')
  parser.add_argument(
      '-p', '--precision', type = str, default = 'fp16', choices=['fp32', 'fp16'], help = 'Weight precision.')
  parser.add_argument(
      '-b', '--batch_size', type = int, default = 200, help = 'Batch size.')
  parser.add_argument(
      '-m', '--max_seq_length', type = int, default = 32, help = 'Max sequence length.')
  parser.add_argument(
      '-a', '--avg_seq_length', type = int, default = 20, help = 'Average sequence length.')
  parser.add_argument(
      '-r', '--real_max_seq_length', type = int, default = 22, help = 'Average sequence length.')
  args = parser.parse_args()
  main(args)
