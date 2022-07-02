import pdb
import tensorflow as tf
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


class ImageCaptioningWithCascadedEncoderDecoder(ImageCaptioning):
    def __init__(self, dataset_name=None):
        super().__init__(dataset_name, 'cascade')

        self.load_dataset(dataset_name)

        self.vocab = self.build_vocabulary(self.caption_data.get_train_captions(),
                                           vocab_type='default',
                                           threshold=10,
                                           prefix=dataset_name)

        self.caption_data.build_caption_seqs(self.vocab, prefix=dataset_name)

        self.embedding = self.build_embedding_matrix('fasttext', self.vocab, prefix=dataset_name)

        self.image_feature_extractor = ImageFeatureExtractorVGG16(self.dataset_name_base,
                                                                  self.caption_data.get_image_dir(),
                                                                  self.workspace_dir, include_top=True)
        self.image_feature_extractor.build_image_features(self.caption_data.get_image_ids())

        print('Building training model ... ', end='')
        self.model = self.build_model()
        print('completed')

        self.callback = SimpleCallback()

    def build_model(self):
        vocab_size = len(self.vocab)
        embedding_dim = self.embedding.get_embedding_dimension()

        inputs1 = Input(shape=(4096,))
        enc = Dropout(0.5)(inputs1)
        enc = Dense(embedding_dim, activation='relu')(enc)
        enc = tf.expand_dims(enc, 1)

        inputs2 = Input(shape=(self.caption_data.get_max_seq_len(),))
        dec = Embedding(vocab_size, embedding_dim,
                         weights=[self.embedding.get_embedding_matrix()],
                         trainable=False,
                         mask_zero=True)(inputs2)
        dec = Dropout(0.5)(dec)
        dec = LSTM(embedding_dim)(tf.concat([enc, dec], 1))
        dec = Dense(embedding_dim, activation='relu')(dec)
        outputs = Dense(vocab_size, activation='softmax')(dec)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model

    def data_loader(self, caption_data, num_images_per_batch, steps_per_epoch):
        X1, X2, y = list(), list(), list()
        n = 0
        while True:
            for image_id in caption_data.train_images:
                n += 1
                image_feature = self.image_feature_extractor.get_feature(image_id)
                for seq in caption_data.caption_seqs[image_id]:
                    for i in range(1, len(seq)):
                        if seq[i] == self.vocab.PAD:
                            break
                        in_seq, out_seq = seq[:i], seq[i]
                        maxlen = self.caption_data.get_max_seq_len() 
                        in_seq = pad_sequences([in_seq], padding='post', maxlen=maxlen)[0]
                        out_seq = to_categorical([out_seq], num_classes=len(self.vocab))[0]
                        X1.append(image_feature)
                        X2.append(in_seq)
                        y.append(out_seq)
 
                if n == num_images_per_batch:
                    yield([np.array(X1), np.array(X2)], np.array(y))
                    X1, X2, y = list(), list(), list()
                    n = 0

    def train(self, total_epochs, num_images_per_batch=3, save_model=False):
        completed_epoch = self.callback.completed_epoch
        if completed_epoch >= total_epochs:
            print('Already completed {} epochs'.format(completed_epoch))
            return

        num_images = len(self.caption_data.train_images)
        steps_per_epoch = num_images // num_images_per_batch

        data_generator = self.data_loader(self.caption_data,
                                          num_images_per_batch,
                                          steps_per_epoch)

        self.model.fit(data_generator,
                       epochs=total_epochs,
                       callbacks=[self.callback],
                       initial_epoch=completed_epoch,
                       steps_per_epoch=steps_per_epoch,
                       verbose=1)

        if save_model:
            # read self.callback.completed_epoch again
            completed_epoch = self.callback.completed_epoch
            self.save_model(completed_epoch)

    def predict(self, image_id):
        maxlen = self.caption_data.get_max_seq_len()
        input_seq = pad_sequences([[self.vocab.START]], padding='post', maxlen=maxlen)
        image_features = self.image_feature_extractor.get_feature(image_id)

        k = 1
        while k < maxlen:
            x1 = tf.convert_to_tensor([image_features])
            x2 = input_seq
            yhat = self.model.predict([x1, x2], verbose=0)
            yhat = np.argmax(yhat)
            if yhat == self.vocab.END:
                break
            else:
                input_seq[0][k] = yhat
                k += 1

        return self.vocab.convert_seq_to_caption(input_seq[0][1:k])
