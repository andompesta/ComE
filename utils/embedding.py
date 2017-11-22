__author__ = 'ando'
import itertools
import logging as log
import math
import numpy as np
from scipy.special import expit as sigmoid

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.INFO)
#
# try:
#     from utils.training_sdg_inner import train_sg, FAST_VERSION
#     log.info('Fast version ' + str(FAST_VERSION))
# except ImportError as e:
#     log.error(e)
#     def train_sg(py_node_embedding, py_context_embedding, py_path, py_alpha, py_negative, py_window, py_table,
#                           py_centroid, py_inv_covariance_mat, py_pi, py_k, py_covariance_mat,
#                           py_lambda1=1.0, py_lambda2=0.0, py_size=None, py_work=None, py_work_o3=None, py_work1_o3=None, py_work2_o3=None,
#                           py_is_node_embedding=1):
#         """
#         Update skip-gram model by training on a single path.
#
#         The path is a list of Vocab objects (or None, where the corresponding
#         word is not in the vocabulary. Called internally from `Word2Vec.train()`.
#
#         This is the non-optimized, Python version.
#         """
#         # loss = 0
#         for pos, node in enumerate(py_path):  # node = input vertex of the system
#             if node is None:
#                 continue  # OOV node in the input path => skip
#
#             labels = np.zeros(py_negative + 1)
#             labels[0] = 1.0  # frist node come from the path, the other not (lable[1:]=0)
#
#             start = max(0, pos - py_window)
#             # now go over all words from the (reduced) window, predicting each one in turn
#             for pos2, node2 in enumerate(py_path[start: pos + py_window + 1], start):  # node 2 are the output nodes predicted form node
#
#
#                 # don't train on OOV words and on the `word` itself
#                 if node2 and not (pos2 == pos):
#                     positive_node_embedding = py_node_embedding[node2.index]  # correct node embeddings
#                     py_work = np.zeros(positive_node_embedding.shape)  # reset the error for each new sempled node
#                     '''
#                     perform negative sampling
#                     '''
#                     # use this word (label = 1) + `negative` other random words not from this path (label = 0)
#                     word_indices = [node.index]
#                     while len(word_indices) < py_negative + 1:
#                         w = py_table[np.random.randint(py_table.shape[0])]  # sample a new negative node
#                         if w != node.index and w != node2.index:
#                             word_indices.append(w)
#
#                     negative_nodes_embedding = py_context_embedding[word_indices]
#
#                     # SGD 1
#                     py_sgd1 = gradient_update(positive_node_embedding, negative_nodes_embedding, labels, py_alpha)
#                     py_work += np.dot(py_sgd1, negative_nodes_embedding)
#
#                     # SDG 2
#                     community_embedding_sgd = 0
#                     if py_lambda2 > 0:
#                         community_embedding_sgd = community_sdg(py_node_embedding, py_centroid, py_inv_covariance_mat,
#                                                                 py_pi,py_k, py_alpha, py_lambda2, node2.index, py_covariance_mat)
#                     # update context
#                     if py_is_node_embedding == 0:
#                         py_context_embedding[word_indices] += np.outer(py_sgd1, positive_node_embedding) # Update context embeddings, note sure if needed in first order
#
#                     # update node embedding
#                     py_node_embedding[node2.index] += (py_lambda1 * py_work) + community_embedding_sgd # Update node embeddings
#
#         return len([word for word in py_path if word is not None])


#sdg gradient update
def gradient_update(positive_node_embedding, negative_nodes_embedding, neg_labels, _alpha):
    '''
      Perform stochastic gradient descent of the first and second order embedding.
      NOTE: using the cython implementation (fast_community_sdg_X) is much more fast
    '''
    fb = sigmoid(np.dot(positive_node_embedding, negative_nodes_embedding.T))  #  propagate hidden -> output
    gb = (neg_labels - fb) * _alpha# vector of error gradients multiplied by the learning rate
    return gb


def community_sdg(node_embedding, centroid, inv_covariance_mat, pi, k, _alpha, _lambda2, index, covariance_mat):
    '''
      Perform stochastic gradient descent of the community embedding.
      NOTE: using the cython implementation (fast_community_sdg_X) is much more fast
    '''
    grad = np.zeros(node_embedding[index].shape, dtype=np.float32)
    # node_loss =  0
    for com in range(k):
        diff = (node_embedding[index] - centroid[com])
        m = pi[index, com] * inv_covariance_mat[com]
        grad += np.dot(m, diff) * _lambda2

    return - np.clip((grad), -0.1 * _alpha, 0.1 * _alpha)
        # node_loss += pi[index, com] * np.log(multivariate_normal.pdf(node_embedding[index], centroid[com], covariance_mat[com]))
    # return - np.clip((grad), -0.1*_alpha, 0.1*_alpha), node_loss


def chunkize_serial(iterable, chunksize, as_numpy=False):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).

    >>> print(list(grouper(range(10), 3)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """

    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[np.array(doc) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        # memory opt: wrap the chunk and then pop(), to avoid leaving behind a dangling reference
        yield wrapped_chunk.pop()

def prepare_sentences(model, paths):
    '''
    :param model: current model containing the vocabulary and the index
    :param paths: list of the random walks. we have to translate the node to the appropriate index and apply the dropout
    :return: generator of the paths according to the dropout probability and the correct index
    '''
    for path in paths:
        # avoid calling random_sample() where prob >= 1, to speed things up a little:
        sampled = [model.vocab[node] for node in path
                   if node in model.vocab and (model.vocab[node].sample_probability >= 1.0 or model.vocab[node].sample_probability >= np.random.random_sample())]
        yield sampled

def batch_generator(iterable, batch_size=1):
    '''
    same as chunkize_serial, but without the usage of an infinite while
    :param iterable: list that we want to convert in batches
    :param batch_size: batch size
    '''
    args = [iterable] * batch_size
    return itertools.zip_longest(*args, fillvalue=None)

class RepeatCorpusNTimes():
    def __init__(self, corpus, n):
        '''
        Class used to repeat n-times the same corpus of paths
        :param corpus: list of paths that we want to repeat
        :param n: number of times we want to repeat our corpus
        '''
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            for document in self.corpus:
                yield document



class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


def xavier_normal(size, as_type=np.float32, gain=1):
    assert len(size) == 2
    std = gain * math.sqrt(2.0 / sum(size))
    return np.random.normal(size=size, loc=0, scale=std).astype(as_type)