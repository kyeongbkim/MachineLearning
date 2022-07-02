import pdb
import sys, os, time

import tensorflow as tf
import pandas as pd
import numpy as np

from collections import OrderedDict

from tensorflow.keras.utils import to_categorical
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import Add
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences

from image_captioning import CaptionData_Flickr8k
from image_captioning import CaptionData_Flickr30k
from image_captioning import ImageCaptioning
from image_captioning import ImageFeatureExtractorVGG16
from image_captioning import SimpleCallback


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def positional_encoding_2d(row,col,d_model):
    assert d_model % 2 == 0
    # first d_model/2 encode row embedding and second d_model/2 encode column embedding
    row_pos = np.repeat(np.arange(row),col)[:,np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col),0),row,axis=0).reshape(-1,1)
    angle_rads_row = get_angles(row_pos,np.arange(d_model//2)[np.newaxis,:],d_model//2)
    angle_rads_col = get_angles(col_pos,np.arange(d_model//2)[np.newaxis,:],d_model//2)
    #apply sin and cos to odd and even indices resp.
    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos_encoding = np.concatenate([angle_rads_row,angle_rads_col],axis=1)[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)  #adding -Inf where mask is 1 s.t. value get ignored in softmax

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask=None):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask=None, padding_mask=None):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    # using look ahead mask so that during self attention current query dont consider future token
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    # use padding mask to avoid padded values of both enc_output and dec_input
    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, row_size,col_size,rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(self.d_model,activation='relu')
        self.pos_encoding = positional_encoding_2d(row_size,col_size, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        # shape(x) = (batch_size,seq_len(H*W),features)
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len(H*W), d_model)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, embedding_matrix=None):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model,
                                                   weights=[embedding_matrix],
                                                   trainable=False,
                                                   mask_zero=True)
        self.pos_encoding = positional_encoding_1d(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,row_size,col_size,
                 target_vocab_size,max_pos_encoding, rate=0.1, embedding_matrix=None):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,row_size,col_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size,max_pos_encoding, rate, embedding_matrix)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def create_masks_decoder(self, target):
        look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
        dec_target_padding_mask = create_padding_mask(target)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return combined_mask

    def call(self, inputs, training=False):
        inp, target, look_ahead_mask = inputs

        enc_padding_mask = None
        dec_padding_mask = None

        enc_output = self.encoder(inp, training, enc_padding_mask)
        # enc_output shape == (batch_size, inp_seq_len, d_model)

        dec_output, attention_weights = self.decoder(target, enc_output, training, look_ahead_mask, dec_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)

        final_output = self.final_layer(dec_output)
        # final_output shape == (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def train_step(self, data):
        inputs, target = data
        img_tensor, _, _ = inputs

        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]

        dec_mask = self.create_masks_decoder(tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self([img_tensor, tar_inp, dec_mask], training=True)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

        return { 'loss': self.train_loss.result(), 'accuracy':self.train_accuracy.result() }


class CustomCallback(SimpleCallback):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.transformer.train_loss.reset_states()
        self.transformer.train_accuracy.reset_states()


class ImageCaptioningWithTransformer(ImageCaptioning):
    def __init__(self, dataset_name=None, batch_size=64):
        super().__init__(dataset_name, 'transformer')

        self.batch_size = batch_size

        self.load_dataset(dataset_name)

        self.vocab = self.build_vocabulary(self.caption_data.get_train_captions(),
                                           vocab_type='default',
                                           threshold=10,
                                           prefix=dataset_name)

        self.caption_data.build_caption_seqs(self.vocab, prefix=dataset_name)

        self.embedding = self.build_embedding_matrix('fasttext', self.vocab, prefix=dataset_name)

        self.image_feature_extractor = ImageFeatureExtractorVGG16(self.dataset_name_base,
                                                                  self.caption_data.get_image_dir(),
                                                                  self.workspace_dir, include_top=False)
        self.image_feature_extractor.build_image_features(self.caption_data.get_image_ids())

        print('Building training model ... ', end='')
        self.model = self.build_model()
        print('completed')

        self.callback = CustomCallback(self.model)

    def build_model(self):
        embedding_matrix = self.embedding.get_embedding_matrix()
        self.embedding_dim = embedding_matrix.shape[-1]

        self.num_layers = 6
        self.d_model = self.embedding_dim
        self.dff = 2048
        self.num_heads = 6
        self.row_size = 8
        self.col_size = 8
        self.target_vocab_size = len(self.vocab)
        self.dropout_rate = 0.1

        model = Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
                            self.row_size, self.col_size, self.target_vocab_size,
                            max_pos_encoding=self.target_vocab_size,
                            rate=self.dropout_rate,
                            embedding_matrix=embedding_matrix)

        learning_rate = CustomSchedule(self.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model.compile(optimizer=optimizer)

        image_feature_shape = self.image_feature_extractor.get_feature_shape()
        maxlen = self.caption_data.get_max_seq_len()

        img_tensor_shape = (self.batch_size,) + image_feature_shape
        target_shape = (self.batch_size, maxlen)
        look_ahead_mask_shape = (self.batch_size, 1, maxlen, maxlen)
        model.build([ img_tensor_shape, target_shape, look_ahead_mask_shape ])
        return model

    def data_loader(self, caption_data, batch_size):
        while True:
            img_tensor, target = list(), list()
            maxlen = caption_data.get_max_seq_len()
            batch = 0
            n = 0
            for image_id, caption_id in caption_data.shuffled_train_data:
                image_feature = self.image_feature_extractor.get_feature(image_id)
                img_tensor.append(image_feature)

                seq = caption_data.caption_seqs[image_id][caption_id]
                padded_seq = pad_sequences([seq], padding='post', maxlen=maxlen)[0]
                target.append(padded_seq)

                n += 1
                if n == batch_size:
                    yield([np.array(img_tensor), np.array(target), None], np.array(target))
                    img_tensor, target = list(), list()
                    n = 0
                    batch += 1

    def train(self, total_epochs, batch_size=64, save_model=False):
        completed_epoch = self.callback.completed_epoch
        if completed_epoch >= total_epochs:
            print('Already completed {} epochs'.format(completed_epoch))
            return

        batch_size = self.batch_size
        steps_per_epoch = len(self.caption_data.shuffled_train_data) // batch_size
        data_generator = self.data_loader(self.caption_data, batch_size)
        self.model.fit(data_generator,
                       epochs=total_epochs,
                       callbacks=[self.callback],
                       initial_epoch=completed_epoch,
                       steps_per_epoch=steps_per_epoch,
                       verbose=1)

    def predict(self, image_id):
        vocab = self.vocab
        max_length = 100

        img_tensor = np.array([ self.image_feature_extractor.get_feature(image_id) ])
        decoder_input = [ vocab.START ]
        output = tf.expand_dims(decoder_input, 0) #tokens
        result = []

        for i in range(max_length):
            dec_mask = self.model.create_masks_decoder(output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.model([img_tensor, output, dec_mask], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == vocab.END:
                break

            # concatentate the predicted_id to the output which is given to the decoder as its input.
            result.append(int(predicted_id))
            output = tf.concat([output, predicted_id], axis=-1)

        return vocab.convert_seq_to_caption(result)

