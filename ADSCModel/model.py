__author__ = 'ando'
import numpy as np
import pickle
from os.path import exists, join as path_join
from os import makedirs
from utils.embedding import Vocab, xavier_normal
from utils.IO_utils import load_ground_true
import logging as log

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)


class Model(object):
    '''
    class that keep track of all the parameters used during the learning of the embedding.
    '''

    def __init__(self, nodes_degree=None,
                 size=2,
                 down_sampling=0,
                 seed=1,
                 table_size=100000000,
                 path_labels='data/',
                 input_file=None):
        '''
        :param nodes_degree: Dict with node_id: degree of node
        :param size: projection space
        :param down_sampling: perform down_sampling of common node
        :param seed: seed for random function
        :param table_size: size of the negative table to generate
        :param path_labels: location of the file containing the ground true (label for each node)
        :param input_file: name of the file containing the ground true (label for each node)
        :return:
        '''

        self.down_sampling = down_sampling
        self.seed = seed
        self.table_size = table_size
        if size % 4 != 0:
            log.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.layer1_size = int(size)

        if nodes_degree is not None:
            self.build_vocab_(nodes_degree)
            self.ground_true, self.k = load_ground_true(path=path_labels, file_name=input_file)
            # inizialize node and context embeddings
            self.make_table()
            self.precalc_sampling()
            self.reset_weights()
        else:
            log.warning("Model not initialized, need the nodes degree")

    def build_vocab_(self, nodes_degree):
        """
        Build vocabulary from a sequence of paths (can be a once-only generator stream).
        Sorted by node id
        """
        # assign a unique index to each word
        self.vocab = {}

        for node_idx, (node, count) in enumerate(sorted(nodes_degree.items(), key=lambda x: x[0])):
            v = Vocab()
            v.count = count
            v.index = node_idx
            self.vocab[node] = v
        self.vocab_size = len(self.vocab)
        log.info("total {} nodes".format(self.vocab_size))

    def precalc_sampling(self):
        '''
            Peach vocabulary item's threshold for sampling
        '''

        if self.down_sampling:
            log.info("frequent-node down sampling, threshold %g; progress tallies will be approximate" % (self.down_sampling))
            total_nodes = sum(v.count for v in self.vocab.values())
            threshold_count = float(self.down_sampling) * total_nodes

        for v in self.vocab.values():
            prob = (np.sqrt(v.count / threshold_count) + 1) * (threshold_count / v.count) if self.down_sampling else 1.0
            v.sample_probability = min(prob, 1.0)

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        np.random.seed(self.seed)
        self.node_embedding = xavier_normal(size=(self.vocab_size, self.layer1_size), as_type=np.float32)
        self.context_embedding = xavier_normal(size=(self.vocab_size, self.layer1_size), as_type=np.float32)


        self.centroid = np.zeros((self.k, self.layer1_size), dtype=np.float32)
        self.covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.inv_covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.pi = np.zeros((self.vocab_size, self.k), dtype=np.float32)

    def reset_communities_weights(self, k):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        np.random.seed(self.seed)
        self.k = k
        self.centroid = np.zeros((self.k, self.layer1_size), dtype=np.float32)
        self.covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.inv_covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.pi = np.zeros((self.vocab_size, self.k), dtype=np.float32)
        log.info("reset communities data| k: {}".format(self.k))


    def reset_weight_random(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        self.node_embedding = np.random.uniform(low=-4.5, high=4.5, size=(self.vocab_size, self.layer1_size)).astype(
            np.float32)
        self.context_embedding = np.random.uniform(low=-4.5, high=4.5, size=(self.vocab_size, self.layer1_size)).astype(np.float32)

        self.centroid = np.zeros((self.k, self.layer1_size), dtype=np.float32)
        self.covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.inv_covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.pi = np.zeros((self.vocab_size, self.k), dtype=np.float32)
        log.info("reset communities data| k: {}".format(self.k))

    def reset_weight_zero(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        self.node_embedding = np.random.uniform(low=-0.5, high=0.5, size=(self.vocab_size, self.layer1_size)).astype(
            np.float32)
        self.context_embedding = np.zeros((self.vocab_size, self.layer1_size), dtype=np.float32)

        self.centroid = np.zeros((self.k, self.layer1_size), dtype=np.float32)
        self.covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.inv_covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.pi = np.zeros((self.vocab_size, self.k), dtype=np.float32)
        log.info("reset communities data| k: {}".format(self.k))

    def make_table(self, power=0.75):
        """
        Create a table using stored vocabulary word counts for drawing random words in the negative
        sampling training routines.

        Called internally from `build_vocab()`.

        """
        log.info("constructing a table with noise distribution from {} words of size {}".format(self.vocab_size, self.table_size))
        # table (= list of words) of noise distribution for negative sampling
        self.table = np.zeros(self.table_size, dtype=np.uint32)
        sorted_keys = sorted(self.vocab.keys())
        k_idx = 0
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        # go through the whole table and fill it up with the word indexes proportional to a word's count**power
        node_idx = sorted_keys[k_idx]
        # normalize count^0.75 by Z
        d1 = self.vocab[node_idx].count**power / train_words_pow
        for tidx in range(self.table_size):
            self.table[tidx] = self.vocab[node_idx].index
            if 1.0 * tidx / self.table_size > d1:
                k_idx += 1
                if k_idx > sorted_keys[-1]:
                    k_idx = sorted_keys[-1]
                node_idx = sorted_keys[k_idx]
                d1 += self.vocab[node_idx].count**power / train_words_pow

        log.info('Max value in the negative sampling table: {}'.format(max(self.table)))



    def save(self, file_name, path='data'):
        if not exists(path):
            makedirs(path)

        with open(path_join(path, file_name + '.bin'), 'wb') as file:
            pickle.dump(self.__dict__, file)

    @staticmethod
    def load_model(file_name, path='data'):
        with open(path_join(path, file_name + '.bin'), 'rb') as file:
            model = Model()
            model.__dict__ = pickle.load(file)
            log.info('model loaded , size: %d \t table_size: %d \t down_sampling: %.5f \t communities %d' %
                  (model.layer1_size, model.table_size, model.down_sampling, model.k))
            return model
