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
import numpy as np
import os
import tensorflow as tf
from bert.modeling import BertConfig, create_attention_mask_from_input_mask
from bert.tokenization import FullTokenizer
from model_layers import EmbeddingLayer, TransformerLayer, EffectiveTransformerLayer, LanguageModelOutputLayer

# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_transformer_weights(checkpoint_path, bert_config, batch_size, max_seq_len, tf_dtype):
    """Load transformer weights"""
    input_embedding_placeholder = tf.placeholder(shape=[batch_size, max_seq_len, bert_config.hidden_size], dtype=tf_dtype, name="input_embedding")
    input_mask_placeholder = tf.placeholder(shape=[batch_size, max_seq_len], dtype=tf.int32, name="input_mask")

    TransformerLayer(bert_config, input_embedding_placeholder, input_mask_placeholder)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        transformer_vars = [v.name for v in all_vars if 'layer' in v.name]
        weights_value = sess.run(transformer_vars)

    tf.reset_default_graph()

    return weights_value


def process_data(batch_size, max_seq_len, tokenizer):
    """process input cloze sentences"""
    input_text = ["What a _ day !", "Effective transformer is _ fast !"]

    # tokenize
    raw_tokens = [['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]'] for text in input_text]

    # mask blanks to fill
    to_predict = [[] for _ in range(len(input_text))]
    for i in range(len(raw_tokens)):
        for j in range(len(raw_tokens[i])):
            if raw_tokens[i][j] == '_':
                raw_tokens[i][j] = '[MASK]'
                to_predict[i].append(j)

    # padding
    input_tokens = [tokens[:max_seq_len] for tokens in raw_tokens]
    input_mask = [[1] * len(tokens) + [0] * (max_seq_len - len(tokens)) for tokens in input_tokens]
    input_tokens = [tokens + ['[PAD]'] * (max_seq_len - len(tokens)) for tokens in input_tokens]

    # tokens to ids
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in input_tokens]

    input_ids += [input_ids[0] for _ in range((batch_size - len(input_ids)))]
    input_mask += [input_mask[0] for _ in range((batch_size - len(input_mask)))]

    return input_ids, input_mask, input_text, to_predict


def main(args):
    checkpoint_path = os.path.join(args.model_dir, "bert_model.ckpt")

    bert_config = BertConfig.from_json_file(os.path.join(args.model_dir, "bert_config.json"))
    bert_config.hidden_dropout_prob = 0.0
    bert_config.attention_probs_dropout_prob = 0.0

    batch_size = args.batch_size
    max_seq_len = args.max_seq_length
    tf_dtype = tf.float16 if args.precision == 'fp16' else tf.float32

    if args.effective_mode:
        # load transformer weights *before* building the computation graph
        weights_value = load_transformer_weights(checkpoint_path, bert_config, batch_size, max_seq_len, tf_dtype)

    # build model
    input_ids_placeholder = tf.placeholder(shape=[batch_size, max_seq_len], dtype=tf.int32, name="input_ids")
    input_mask_placeholder = tf.placeholder(shape=[batch_size, max_seq_len], dtype=tf.int32, name="input_mask")
    attention_mask_placeholder = tf.placeholder(shape=[batch_size, max_seq_len, max_seq_len], dtype=tf_dtype, name="attention_mask")
    input_embedding_placeholder = tf.placeholder(shape=[batch_size, max_seq_len, bert_config.hidden_size], dtype=tf_dtype, name="input_embedding")
    embedding_table_placeholder = tf.placeholder(shape=[bert_config.vocab_size, bert_config.hidden_size], dtype=tf_dtype, name="embedding_table")
    transformer_output_placeholder = tf.placeholder(shape=[batch_size, max_seq_len, bert_config.hidden_size], dtype=tf_dtype, name="transformer_output")

    embedding_layer = EmbeddingLayer(bert_config, input_ids_placeholder)
    if args.effective_mode:
        effective_transformer_layer = EffectiveTransformerLayer(batch_size, max_seq_len, bert_config,
                                                                attention_mask_placeholder, input_mask_placeholder,
                                                                input_embedding_placeholder, weights_value)
    else:
        standard_transformer_layer = TransformerLayer(bert_config, input_embedding_placeholder, input_mask_placeholder)
    output_layer = LanguageModelOutputLayer(bert_config, transformer_output_placeholder, embedding_table_placeholder)

    # model saver
    variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(variables_to_restore)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        # restore embedding layer and output layer
        saver.restore(sess, checkpoint_path)

        # process input data
        tokenizer = FullTokenizer(vocab_file=os.path.join(args.model_dir, 'vocab.txt'))
        input_ids, input_mask, input_text, to_predict = process_data(batch_size, max_seq_len, tokenizer)
        input_ids_tensor = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        input_mask_tensor = tf.convert_to_tensor(input_mask, dtype=tf.int32)

        # predict
        begin = datetime.now()
        input_embedding, embedding_table = sess.run(
            [embedding_layer.get_embedding_output(), embedding_layer.get_embedding_table()],
            feed_dict={input_ids_placeholder: input_ids})
        attention_mask = sess.run(create_attention_mask_from_input_mask(input_ids_tensor, input_mask_tensor))
        if args.effective_mode:
            transformer_output = sess.run(effective_transformer_layer.get_transformer_output(),
                                          feed_dict={input_embedding_placeholder: input_embedding,
                                                     attention_mask_placeholder: attention_mask,
                                                     input_mask_placeholder: input_mask})
        else:
            transformer_output = sess.run(standard_transformer_layer.get_transformer_output(),
                                          feed_dict={input_embedding_placeholder: input_embedding,
                                                     attention_mask_placeholder: attention_mask,
                                                     input_mask_placeholder: input_mask})
        probs = sess.run(output_layer.get_predict_probs(),
                         feed_dict={transformer_output_placeholder: transformer_output,
                                    embedding_table_placeholder: embedding_table})
        end = datetime.now()
        print("time cost: ", (end - begin).total_seconds(), "s")

        # choose top k answers
        k = 5
        top_ids = np.argsort(-probs, axis=2)[:, :, :k]

        batch_results = []
        for sid, blank_ids in enumerate(to_predict):
            sentence_results = []
            for cid in blank_ids:
                result = []
                for idx in top_ids[sid][cid]:
                    token = tokenizer.convert_ids_to_tokens([idx])[0]
                    result.append((token, probs[sid][cid][idx]))
                sentence_results.append(result)
            batch_results.append(sentence_results)

    for text, blank_ids, sentence_results in zip(input_text, to_predict, batch_results):
        print("Q:", text)
        for cid, result in zip(blank_ids, sentence_results):
            print("A:", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bert performance measuring sample.')
    parser.add_argument('-d', '--model_dir', type=str, required=True, help='Model directory.')
    parser.add_argument('-e', '--effective_mode', action='store_true', help='Create model with effective transformer.')
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['fp32', 'fp16'], help='Weight precision.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('-m', '--max_seq_length', type=int, default=32, help='Max sequence length.')
    args = parser.parse_args()
    main(args)
