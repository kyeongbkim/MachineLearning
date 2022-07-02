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


class Encoder(tf.keras.Model):
    # This encoder passes the features through a Fully connected layer
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')

    def call(self, x):
        x = self.fc(x)
        return x    


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, embedding_matrix, batch_size):
        super().__init__()
        self.units = units
        self.batch_size = batch_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False,
                                                   mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       stateful=True)
        self.gru.build((batch_size, 1, 2*embedding_dim))
    
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
        self.batchnormalization = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                beta_initializer='zeros', gamma_initializer='ones',
                moving_mean_initializer='zeros', moving_variance_initializer='ones',
                beta_regularizer=None, gamma_regularizer=None,
                beta_constraint=None, gamma_constraint=None)

        self.fc2 = tf.keras.layers.Dense(vocab_size)

        # Implementing Attention Mechanism 
        self.Uattn = tf.keras.layers.Dense(units)
        self.Wattn = tf.keras.layers.Dense(units)
        self.Vattn = tf.keras.layers.Dense(1)

    def call(self, x, features, hidden):
        # batch_size = 64, encoder_attention_dim=49, embedding_dim=300, hidden_size=512
        # x shape == (64,1)
        # features shape == (64, 49, 512) ==> Output from ENCODER
        # hidden shape == (64, 512)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # hidden_with_time_axis shape == (64,1,512)

        # Attention Function
        # e(ij) = f(s(t-1),h(j))
        # e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))
        # self.Uattn(features) : (64,49,512)
        # self.Wattn(hidden_with_time_axis) : (64,1,512)
        # tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
        # self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score
        # you get 1 at the last axis because you are applying score to self.Vattn
        score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))
        # score shape == (64, 49, 1)

        # Then find Probability using Softmax
        '''attention_weights(alpha(ij)) = softmax(e(ij))'''
        attention_weights = tf.nn.softmax(score, axis=1)
        # attention_weights shape == (64, 49, 1)

        # Give weights to the different pixels in the image
        ''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) ''' 
        context_vector = attention_weights * features
        # ContextVector(64,49,300) = AttentionWeights(64,49,1) * features(64,49,300)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # context_vector shape after sum == (64,300)

        x = self.embedding(x)
        # x shape after passing through embedding == (64, 1, 300)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # x shape after concatenation == (64, 1, 600)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        x = self.fc1(output)
        # x shape == (64,1,512)

        x = tf.reshape(x, (-1, x.shape[2]))
        # x shape == (64,512)

        # Adding Dropout and BatchNorm Layers
        x= self.dropout(x)
        x= self.batchnormalization(x)
        # output shape == (64,512)
        x = self.fc2(x)
        # shape == (64, vocab_size)
        return x, state, attention_weights

    def reset_states(self):
        self.gru.reset_states()
        return tf.zeros((self.batch_size, self.units))


class Attention(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, maxlen, embedding_matrix, batch_size):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim, units, vocab_size, embedding_matrix, batch_size)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def loss_function(self, real, pred):
        loss_ = self.loss_object(real, pred)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_), tf.reduce_sum(mask)

    def call(self, inputs):
        img_tensor, target = inputs
        sum_loss = 0.
        sum_mask = 0.

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_states()
        dec_input = tf.expand_dims(target[:, 0], 1)

        features = self.encoder(img_tensor)
        for i in range(1, self.maxlen):
            # passing the features through the decoder
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            loss_, mask_ = self.loss_function(target[:, i], predictions)
            sum_loss += loss_
            sum_mask += mask_
   
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

        return predictions, sum_loss/sum_mask

    def train_step(self, data):
        inputs, target = data

        with tf.GradientTape() as tape:
            predictions, loss = self(inputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)

        return { 'loss': self.train_loss.result() }


class CustomCallback(SimpleCallback):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.transformer.train_loss.reset_states()


class ImageCaptioningWithAttention(ImageCaptioning):
    def __init__(self, dataset_name=None, batch_size=64):
        super().__init__(dataset_name, 'attention')

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
        self.units = 512
        maxlen = self.caption_data.get_max_seq_len()
        model = Attention(self.embedding_dim, self.units, len(self.vocab), maxlen, embedding_matrix, self.batch_size)
        model.compile(optimizer='adam')

        image_feature_shape = self.image_feature_extractor.get_feature_shape()
        img_tensor_shape = (self.batch_size,) + image_feature_shape
        target_shape = (self.batch_size, maxlen)
        model.build([img_tensor_shape, target_shape])

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
                    yield([np.array(img_tensor), np.array(target)], np.array(target))
                    img_tensor, target = list(), list()
                    n = 0
                    batch += 1

    def train(self, total_epochs):
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
        batch_size = self.batch_size
        max_length = 100

        img_tensor = np.array([ self.image_feature_extractor.get_feature(image_id) ])
        dec_input = tf.expand_dims([vocab.START], 1)
        features = self.model.encoder(img_tensor)
        hidden = self.model.decoder.reset_states()
        result = []
        for i in range(max_length):
            dummy_dec_input = tf.zeros([batch_size-1, dec_input.shape[1]], dtype=dec_input.dtype)
            dec_input_batch = tf.concat([dec_input, dummy_dec_input], 0)
            dummy_features = tf.zeros([batch_size-1, features.shape[1], features.shape[2]], dtype=features.dtype)
            features_batch = tf.concat([features, dummy_features], 0)

            predictions, hidden, attention_weights = self.model.decoder(dec_input_batch, features_batch, hidden)
            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(int(predicted_id))
 
            if predicted_id == vocab.END:
                break

            dec_input = tf.expand_dims([predicted_id], 0)
 
        return vocab.convert_seq_to_caption(result)

