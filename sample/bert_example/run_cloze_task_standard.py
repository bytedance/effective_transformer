# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2020/6/1 12:28
@Description:
"""
from datetime import datetime
import argparse
import numpy as np
import os
import tensorflow as tf
from bert.modeling import BertConfig, create_attention_mask_from_input_mask
from bert.tokenization import FullTokenizer
from model_layers import EmbeddingLayer, TransformerLayer, LanguageModelOutputLayer
from run_cloze_task_effective import process_data

# disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(args):
    checkpoint_path = os.path.join(args.model_dir, "bert_model.ckpt")

    bert_config = BertConfig.from_json_file(os.path.join(args.model_dir, "bert_config.json"))
    bert_config.hidden_dropout_prob = 0.0
    bert_config.attention_probs_dropout_prob = 0.0

    batch_size = args.batch_size
    max_seq_len = args.max_seq_length

    # build model
    input_ids_placeholder = tf.placeholder(shape=[batch_size, max_seq_len], dtype=tf.int32, name="input_ids")
    input_mask_placeholder = tf.placeholder(shape=[batch_size, max_seq_len], dtype=tf.int32, name="input_mask")
    attention_mask_placeholder = tf.placeholder(shape=[batch_size, max_seq_len, max_seq_len], dtype=tf.float32, name="attention_mask")
    input_embedding_placeholder = tf.placeholder(shape=[batch_size, max_seq_len, bert_config.hidden_size], dtype=tf.float32, name="input_embedding")
    embedding_table_placeholder = tf.placeholder(shape=[bert_config.vocab_size, bert_config.hidden_size], dtype=tf.float32, name="embedding_table")
    transformer_output_placeholder = tf.placeholder(shape=[batch_size, max_seq_len, bert_config.hidden_size], dtype=tf.float32, name="transformer_output")

    embedding_layer = EmbeddingLayer(bert_config, input_ids_placeholder)
    transformer_layer = TransformerLayer(bert_config, input_embedding_placeholder, input_mask_placeholder)
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
        transformer_output = sess.run(transformer_layer.get_transformer_output(),
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
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('-m', '--max_seq_length', type=int, default=32, help='Max sequence length.')
    args = parser.parse_args()
    main(args)
