# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2020/6/1 12:28
@Description:
"""
import tensorflow as tf
from bert import modeling
import effective_transformer


class EmbeddingLayer(object):
    def __init__(self, config, input_ids, token_type_ids=None):
        input_shape = modeling.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        # Keep variable names the same as BERT
        with tf.variable_scope("bert"):
            with tf.variable_scope("embeddings"):
                (embedding_output, self.embedding_table) = modeling.embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=False)

                self.embedding_output = modeling.embedding_postprocessor(
                    input_tensor=embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


class TransformerLayer(object):
    def __init__(self, config, input_embedding, input_mask=None):
        input_shape = modeling.get_shape_list(input_embedding, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        # Keep variable names the same as BERT
        with tf.variable_scope("bert"):
            with tf.variable_scope("encoder"):
                attention_mask = modeling.create_attention_mask_from_input_mask(input_embedding, input_mask)

                all_encoder_layers = modeling.transformer_model(
                    input_tensor=input_embedding,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=modeling.get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

                self.sequence_output = all_encoder_layers[-1]

    def get_transformer_output(self):
        return self.sequence_output


class EffectiveTransformerLayer(object):
    def __init__(self, batch_size, max_seq_len, bert_config, attention_mask_tensor, input_mask_tensor,
                 input_embedding_tensor, weights_value):
        self.sequence_output = effective_transformer.get_sequence_output(
            max_batch_size=batch_size,
            max_seq_length=max_seq_len,
            config=bert_config,
            attention_mask=attention_mask_tensor,
            input_mask=input_mask_tensor,
            from_tensor=input_embedding_tensor,
            weights_value=weights_value,
        )

    def get_transformer_output(self):
        return self.sequence_output


class LanguageModelOutputLayer(object):
    def __init__(self, config, input_hidden, embedding_table):
        # Keep variable names the same as BERT
        with tf.variable_scope("cls"):
            with tf.variable_scope("predictions"):
                with tf.variable_scope("transform"):
                    self.transformed_output = tf.layers.dense(
                        input_hidden,
                        config.hidden_size,
                        activation=modeling.get_activation(config.hidden_act),
                        kernel_initializer=modeling.create_initializer(config.initializer_range))
                    self.transformed_output = modeling.layer_norm(self.transformed_output)

                output_bias = tf.Variable(tf.zeros([config.vocab_size]), name="output_bias")
                self.final_output = tf.add(tf.matmul(self.transformed_output, tf.transpose(embedding_table)),
                                           output_bias)
                self.probs = tf.nn.softmax(self.final_output, name='token_probs')

    def get_predict_probs(self):
        return self.probs
