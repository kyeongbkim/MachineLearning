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
        super(Encoder, self).__init__()
        # shape after fc == (batch_size, 49, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

    def call(self, x):
        #x= self.dropout(x)
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x    


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, embedding_matrix):
        super(Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False,
                                                   mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
    
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
        # features shape ==> (64,49,256) ==> Output from ENCODER

        # hidden shape == (batch_size, hidden_size) ==>(64,512)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size) ==> (64,1,512)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (64, 49, 1)
        # Attention Function
        '''e(ij) = f(s(t-1),h(j))'''
        ''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''
        score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))
        # self.Uattn(features) : (64,49,512)
        # self.Wattn(hidden_with_time_axis) : (64,1,512)
        # tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
        # self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score
        # you get 1 at the last axis because you are applying score to self.Vattn


        # Then find Probability using Softmax
        '''attention_weights(alpha(ij)) = softmax(e(ij))'''
        attention_weights = tf.nn.softmax(score, axis=1)
        # attention_weights shape == (64, 49, 1)


        # Give weights to the different pixels in the image
        ''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) ''' 
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # Context Vector(64,256) = AttentionWeights(64,49,1) * features(64,49,256)
        # context_vector shape after sum == (64, 256)

        # x shape after passing through embedding == (64, 1, 256)
        x = self.embedding(x)

        # x shape after concatenation == (64, 1,  512)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # Adding Dropout and BatchNorm Layers
        x= self.dropout(x)
        x= self.batchnormalization(x)
        # output shape == (64 * 512)
        x = self.fc2(x)
        # shape : (64 * 8329(vocab))
        return x, state, attention_weights

    def reset_state(self, batch_size):
      return tf.zeros((batch_size, self.units))


class ImageCaptioningWithAttention(ImageCaptioning):
    def __init__(self, dataset_name=None):
        super().__init__()

        if dataset_name == 'Flickr8k':
            dataset_name_base = 'Flickr8k'
            self.caption_data = CaptionData_Flickr8k(self.data_dir, self.workspace_dir,
                                                     limit_num_images=8000, train_data_ratio=0.8)
        elif dataset_name.startswith('Flickr30k'):
            dataset_name_base = 'Flickr30k'
            try:
                dataset_args = dataset_name.split('_')
                limit_num_images = int(dataset_args[1])
                train_data_ratio = float(dataset_args[2])
            except:
                limit_num_images = 30000
                train_data_ratio = 0.8

            self.caption_data = CaptionData_Flickr30k(self.data_dir, self.workspace_dir,
                                                      limit_num_images=limit_num_images,
                                                      train_data_ratio=train_data_ratio)
        else:
            assert False, 'Unknown dataset_name {}'.format(dataset_name)

        self.vocab = self.build_vocabulary(self.caption_data.get_captions(),
                                           vocab_type='default',
                                           threshold=10,
                                           prefix=dataset_name)

        self.caption_data.build_caption_seqs(self.vocab, prefix=dataset_name)

        self.embedding = self.build_embedding_matrix('fasttext', self.vocab, prefix=dataset_name)

        self.image_feature_extractor = ImageFeatureExtractorVGG16(dataset_name_base,
                                                                  self.caption_data.get_image_dir(),
                                                                  self.workspace_dir, include_top=False)
        self.image_feature_extractor.build_image_features(self.caption_data.get_image_ids())

        self.set_model_name_prefix('{}-attention'.format(dataset_name))
        self.model = self.load_model()
        if not self.model:
            print('Building training model ... ', end='')
            self.model = self.build_model()
            print('completed')

        self.callback = SimpleCallback()

    def build_model(self):
        embedding_matrix = self.embedding.get_embedding_matrix()
        self.embedding_dim = embedding_matrix.shape[-1]
        self.units = 512

        self.encoder = Encoder(self.embedding_dim)
        self.decoder = Decoder(self.embedding_dim, self.units, len(self.vocab), embedding_matrix)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        return None

    def data_loader(self, caption_data, batch_size):
        img_tensor, target = list(), list()
        total_num_batches = len(caption_data.shuffled_train_data) // batch_size
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
                yield(batch, total_num_batches, np.array(img_tensor), np.array(target))
                img_tensor, target = list(), list()
                n = 0
                batch += 1

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


    @tf.function
    def train_step(self, vocab, img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims(target[:, 0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
   
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)
   
        total_loss = (loss / int(target.shape[1]))
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss, total_loss

    def train(self, total_epochs, batch_size=64, save_model=False):
        completed_epoch = self.callback.completed_epoch
        if completed_epoch >= total_epochs:
            print('Already completed {} epochs'.format(completed_epoch))
            return

        for epoch in range(completed_epoch, total_epochs):
            data_generator = self.data_loader(self.caption_data, batch_size)

            start = time.time()
            total_loss = 0

            print('Epoch {}:'.format(epoch + 1))
            self.callback.on_epoch_begin(epoch=epoch)

            num_steps = 0
            for batch, total_num_batches, img_tensor, target in data_generator:
                batch_loss, t_loss = self.train_step(self.vocab, img_tensor, target)
                total_loss += t_loss
                num_steps += 1

                is_last = (batch + 1) == total_num_batches
                if num_steps % 50 == 0 or is_last:
                    print ('\r    Batch {}/{} Loss {:.4f}'.format(
                               batch + 1, total_num_batches,
                               batch_loss.numpy()/int(target.shape[1])), end='')

            self.callback.on_epoch_end(epoch=epoch, logs={'loss': total_loss/num_steps} )
            print ('\r\n    Epoch {} Loss {:.4f}, Time {:.1f} secs'.format(
                   epoch + 1, total_loss/num_steps, time.time() - start))

        if save_model:
            completed_epoch = self.callback.completed_epoch
            # FIXME
            #self.save_model(completed_epoch)

    def predict(self, image_id):
        vocab = self.vocab
        max_length = 100

        img_tensor = np.array([ self.image_feature_extractor.get_feature(image_id) ])
        dec_input = tf.expand_dims([vocab.START], 1)
        features = self.encoder(img_tensor)
        hidden = self.decoder.reset_state(batch_size=1)
        result = []
        for i in range(max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)
            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(int(predicted_id))
 
            if predicted_id == vocab.END:
                break

            dec_input = tf.expand_dims([predicted_id], 0)
 
        return vocab.convert_seq_to_caption(result)

