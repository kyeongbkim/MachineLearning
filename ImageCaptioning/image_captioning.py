import pdb
import sys
import os
import io
import pandas as pd
import numpy as np
import nltk
import pickle
from collections import Counter, OrderedDict

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import layers, models
from keras.callbacks import Callback
from sklearn.utils import shuffle


class CaptionData(object):
    def __init__(self):
        self.data_dir = None
        self.workspace_dir = None
        self.image_dir = None

        self.captions = None
        self.train_images = None
        self.test_images = None
        self.caption_seqs = None
        self.max_seq_len = 0

    def get_image_dir(self):
        return self.image_dir

    def get_image_ids(self):
        return self.captions.keys()

    def get_train_captions(self):
        captions = []
        for image_id in self.train_images:
            captions.extend(self.captions[image_id])
        return captions

    def get_all_captions(self):
        # flatten the captions
        return [ item for sublist in self.captions.values() for item in sublist]

    def get_captions(self, image_id):
        return self.captions[image_id]

    def build_caption_seqs(self, vocab, f_save=True, prefix='unknown'):
        caption_seqs_file_path = self.workspace_dir + '/{}-caption_sequences.pkl'.format(prefix)
        if os.path.exists(caption_seqs_file_path):
            print('Loading caption sequences {} ... '.format(caption_seqs_file_path), end='', flush=True)
            with open(caption_seqs_file_path, 'rb') as f:
                caption_seqs = pickle.load(f)
            print('completed')
        else:
            caption_seqs = OrderedDict()
            for i, image_id in enumerate(self.captions):
                if image_id not in caption_seqs:
                    caption_seqs[image_id] = []
                for caption in self.captions[image_id]:
                    seq = vocab.convert_caption_to_seq(caption)
                    caption_seqs[image_id].append(seq)
            if f_save:
                with open(caption_seqs_file_path, 'wb') as f:
                    pickle.dump(caption_seqs, f)
        self.caption_seqs = caption_seqs
        self.max_seq_len = max([len(seq) for seq_list in caption_seqs.values() for seq in seq_list])

    def get_max_seq_len(self):
        return self.max_seq_len


class CaptionData_Flickr8k(CaptionData):
    def __init__(self, data_dir, workspace_dir, limit_num_images=0, train_data_ratio=0.8):
        self.data_dir = data_dir
        self.workspace_dir = workspace_dir
        self.image_dir    = data_dir + '/flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/'

        train_images_file = data_dir + '/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
        dev_images_file   = data_dir + '/flickr8k/Flickr8k_text/Flickr_8k.devImages.txt'
        test_images_file  = data_dir + '/flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
        caption_file      = data_dir + '/flickr8k/Flickr8k_text/Flickr8k.token.txt'

        # Limit the number of images
        if limit_num_images <= 0:
            limit_num_images = sys.maxsize

        # build dictionary: image_id-to-captions
        num_images = 0
        self.captions = OrderedDict()
        with open(caption_file, 'r') as infile:
            for line in infile:
                tokens = line.split()
                if len(line) > 2:
                  image_id = tokens[0].split('#')[0]

                  #check if image file exists
                  if not os.path.exists(self.image_dir + '/' + image_id):
                      continue

                  caption = ' '.join(tokens[1:])
                  if image_id not in self.captions:
                      if num_images >= limit_num_images:
                          continue
                      self.captions[image_id] = list()
                      num_images += 1
                  self.captions[image_id].append(caption)

        assert num_images == len(self.captions)
        train_test_boundary = int(num_images * train_data_ratio)

        self.train_images = []
        self.test_images = []
        train_data = []
        for i, image_id in enumerate(self.captions):
            if i >= num_images:
                break
            if i < train_test_boundary:
                self.train_images.append(image_id)
                train_data.extend( [[image_id, caption_id] for caption_id, _ \
                                                           in enumerate(self.captions[image_id])] )
            else:
                self.test_images.append(image_id)
        self.shuffled_train_data = shuffle(train_data, random_state=1)


class CaptionData_Flickr30k(CaptionData):
    def __init__(self, data_dir, workspace_dir, limit_num_images=0, train_data_ratio=0.8):
        self.data_dir = data_dir
        self.workspace_dir = workspace_dir
        self.image_dir = data_dir + '/Flickr30k_dataset/flickr30k_images/flickr30k_images'

        caption_file_path = data_dir + '/Flickr30k_dataset/flickr30k_images/results_fixed.csv'

        self.pd_data = pd.read_csv(caption_file_path, delimiter='|')
        self.pd_data.columns = [ 'image_id', 'caption_id', 'caption' ]
        self.pd_data['image_id'] = self.pd_data['image_id'].str.strip()
        self.pd_data['caption'] = self.pd_data['caption'].str.strip()

        # Limit the number of images
        if limit_num_images <= 0:
            limit_num_images = sys.maxsize

        # build dictionary: image_id-to-captions
        num_images = 0
        self.captions = OrderedDict()
        for idx, row in self.pd_data.iterrows():
            if row.image_id not in self.captions:
                if num_images >= limit_num_images:
                    continue
                self.captions[row.image_id] = []
                num_images += 1
            self.captions[row.image_id].append(row.caption)

        # split data: train vs test
        assert num_images == len(self.captions)
        train_test_boundary = int(num_images * train_data_ratio)

        self.train_images = []
        self.test_images = []
        train_data = []
        for i, image_id in enumerate(self.captions):
            if i >= num_images:
                break
            if i < train_test_boundary:
                self.train_images.append(image_id)
                train_data.extend( [[image_id, caption_id] for caption_id, _ \
                                                           in enumerate(self.captions[image_id])] )
            else:
                self.test_images.append(image_id)
        self.shuffled_train_data = shuffle(train_data, random_state=1)


class Vocabulary(object):
    PAD   = 0
    START = 1
    END   = 2
    UNK   = 3

    """Simple vocabulary wrapper."""
    def __init__(self, captions, threshold=4):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

        self.build_vocab(captions, threshold)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def build_vocab(self, captions, threshold):
        """Build a simple vocabulary wrapper."""
        counter = Counter()
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if (i+1) % 1000 == 0:
                print("\rBuilding vocabulary ... {}/{} tokenized. {}% complete".format(
                    i+1, len(captions), int((i+1)*100/len(captions))), end='', flush=True)
                
        print("\rBuilding vocabulary ... {}/{} tokenized. {}% completed".format(
            i+1, len(captions), int((i+1)*100/len(captions))))

        # If the word frequency is less than 'threshold', then the word is discarded.
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            self.add_word(word)

    def convert_caption_to_seq(self, caption, prepend_start=True, append_end=True):
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_seq = []
        if prepend_start:
            caption_seq.append(self('<start>'))
        caption_seq.extend([self(token) for token in tokens])
        if append_end:
            caption_seq.append(self('<end>'))
        return caption_seq

    def convert_seq_to_caption(self, seq):
        caption = ''
        for idx in seq:
            if idx in [ self.PAD, self.START, self.END, self.UNK ]:
                continue
            caption = caption + ' ' + self.idx2word[idx]
        return caption


class FastTextEmbedding(object):
    def __init__(self, data_dir, vocab):
        self.fasttext_vec_file_path = data_dir + '/wiki-news-300d-1M.vec'

        self.embedding_dim = 300
        vocab_size = len(vocab)

        embedding_vectors = self.load_vectors(self.fasttext_vec_file_path)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        for word, idx in vocab.word2idx.items():
            embedding_vector = embedding_vectors.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = list(embedding_vector)
        self.embedding_matrix = embedding_matrix

    def load_vectors(self, fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())  # n lines in d dimension
        data = {}
        i = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
            i += 1
            if i % 10000 == 0:
                print('\rLoading embedding vectors ... {}/{} loaded. {}% complete'.format(
                    i, n, int(100*i/n)), end='', flush=True)

        print('\rLoading embedding vectors ... {}/{} loaded. {}% completed'.format(
                i, n, int(100*i/n)))
        return data

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def get_embedding_dimension(self):
        return self.embedding_dim


class ImageFeatureExtractorVGG16(object):
    def __init__(self, prefix, image_dir, workspace_dir, include_top=True):
        self.image_dir = image_dir
        self.reshape = None
        if include_top:
            self.feature_dir = workspace_dir + '/' + prefix + '-vgg16-include_top'
        else:
            self.feature_dir = workspace_dir + '/' + prefix + '-vgg16-no_include_top'
            self.reshape = True

        base_model = VGG16(weights="imagenet", include_top=include_top, input_shape=(224,224,3))
        base_model.trainable = False ## Not trainable weights
        output_layer = base_model.layers[-2] if include_top else base_model.layers[-1]
        self.model = tf.keras.Model(base_model.input, output_layer.output)
        if self.reshape:
            self.reshape = (-1, self.model.output.shape[-1])

    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.vgg16.preprocess_input(img)
        return img

    def build_image_features(self, image_ids, force=False):
        self.features = {}

        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir)

        for i, image_id in enumerate(image_ids):
            image_file_path = self.image_dir + '/' + image_id
            feature_file_path = self.feature_dir + '/' + image_id + '.npy'
            if force or not os.path.exists(feature_file_path):
                img = self.load_image(image_file_path)
                features = self.model(tf.expand_dims(img, 0))[0]
                if self.reshape:
                    features = tf.reshape(features, self.reshape)
                np.save(feature_file_path, features.numpy()) # the suffix ".npy" will be appened automatically
            else:
                features = self.get_feature_from_file(image_id)
            self.features[image_id] = features

            if (i+1) % 50 == 0:
                print("\rBuilding image features {} ... {}/{} processed. {}% complete".format(
                    self.feature_dir, i+1, len(image_ids), int((i+1)*100/len(image_ids))), end='', flush=True)

        print("\rBuilding image features {} ... {}/{} processed. {}% completed".format(
            self.feature_dir, i+1, len(image_ids), int((i+1)*100/len(image_ids))))

    def get_feature_from_file(self, image_id):
        feature_file_path = self.feature_dir + '/' + image_id + '.npy'
        if os.path.exists(feature_file_path):
            feature = np.load(feature_file_path)
            return feature
        return None

    def get_feature(self, image_id):
        return self.features[image_id]

    def get_feature_shape(self):
        return next(iter(self.features.values())).shape

class ImageCaptioning(object):
    def __init__(self, dataset_name, model_name):
        self.data_dir = './data'
        self.workspace_dir = './workspace'
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.artifacts_dir= self.workspace_dir + '/' + dataset_name + '-' + model_name

        self.model = None

    def load_dataset(self, dataset_name):
        if dataset_name == 'Flickr8k':
            self.dataset_name_base = 'Flickr8k'
            self.caption_data = CaptionData_Flickr8k(self.data_dir, self.workspace_dir,
                                                     limit_num_images=8000, train_data_ratio=0.8)
        elif dataset_name.startswith('Flickr30k'):
            self.dataset_name_base = 'Flickr30k'
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

    def build_embedding_matrix(self, algorithm, vocab, f_save=True, prefix='unknown'):
        embedding_matrix_file_path = self.workspace_dir + '/{}-embedding_matrix_fasttext.pkl'.format(prefix)

        if os.path.exists(embedding_matrix_file_path):
            print('Loading prebuilt embedding matrix {} ... '.format(embedding_matrix_file_path), end='', flush=True)
            with open(embedding_matrix_file_path, 'rb') as f:
                embedding = pickle.load(f)
            print('completed')
        else:
            embedding = None
            if algorithm == 'fasttext':
                embedding = FastTextEmbedding(self.data_dir, vocab)
            else:
                assert False, 'algorithm {} not supported'.format(algorithm)

            if f_save and embedding:
                    with open(embedding_matrix_file_path, 'wb') as f:
                        pickle.dump(embedding, f)
        return embedding

    def build_vocabulary(self, captions, vocab_type='default', threshold=4, f_save=True, prefix='unknown'):
        vocab_file_path = self.workspace_dir + '/{}-vocab.pkl'.format(prefix)
        if os.path.exists(vocab_file_path):
            print('Loading prebuilt vocabulary {} ... '.format(vocab_file_path), end='', flush=True)
            with open(vocab_file_path, 'rb') as f:
                vocab = pickle.load(f)
            print('completed')
        else:
            vocab = None
            if vocab_type == 'default':
                vocab = Vocabulary(captions, threshold=threshold)
            else:
                assert False, 'vocab type {} not supported'.format(vocab_type)

            if f_save and vocab:
                with open(vocab_file_path, 'wb') as f:
                    pickle.dump(vocab, f)
        return vocab

    def load_weights(self, epoch):
        load_path = self.artifacts_dir + '/ckpt-' + str(epoch)
        print('Loading weights {}'.format(load_path))
        return self.model.load_weights(load_path)

    def save_weights(self, epoch):
        save_path = self.artifacts_dir + '/ckpt-' + str(epoch)
        print('Saving weights {}'.format(save_path))
        self.model.save_weights(save_path)

    def clear_checkpoints(self):
        if os.path.exists(self.artifacts_dir):
            import shutil
            shutil.rmtree(self.artifacts_dir)

    def train(self):
        assert False, 'must be implemented in subclasses'

    def predict(self, image_id):
        assert False, 'must be implemented in subclasses'

    def sample_test(self, image_id, show_image=False):
        references = self.caption_data.get_captions(image_id)
        print('Predicted:', self.predict(image_id))
        print('Reference:')
        for txt in references:
            print('    ', txt)

        if show_image:
            import matplotlib.pyplot as plt
            image_path = self.image_feature_extractor.image_dir + '/' + image_id
            plt.imshow(plt.imread(image_path))
            plt.show()


class SimpleCallback(Callback):
    def __init__(self, completed_epoch=0):
        super().__init__()
        self.completed_epoch = completed_epoch
        self.completed_batch = 0
        self.losses = OrderedDict()

    def on_train_batch_end(self, batch, logs=None):
        #print('on_train_batch_end: batch={}, logs = {}'.format(batch, logs))
        self.completed_batch = batch + 1

    def on_epoch_begin(self, epoch, logs=None):
        #print('on_epoch_begin: epoch={}, logs={}'.format(epoch, logs))
        pass

    def on_epoch_end(self, epoch, logs=None):
        #print('on_epoch_end: epoch {}, logs={}'.format(epoch, logs))
        self.completed_epoch = epoch + 1
        self.completed_batch = 0
        self.losses[self.completed_epoch] = logs['loss']
