__author__ = 'ando'

import numpy as np
from time import time
import logging as log
import random

import networkx as nx
from itertools import zip_longest
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from os import path
from collections import Counter

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.INFO)


def __random_walk__(G, path_length, start, alpha=0, rand=random.Random()):
    '''
    Returns a truncated random walk.
    :param G: networkx graph
    :param path_length: Length of the random walk.
    :param alpha: probability of restarts.
    :param rand: random number generator
    :param start: the start node of the random walk.
    :return:
    '''

    path = [start]


    while len(path) < path_length:
        cur = path[-1]
        if len(G.neighbors(cur)) > 0:
            if rand.random() >= alpha:
                path.append(rand.choice(G.neighbors(cur)))
            else:
                path.append(path[0])
        else:
            break
    return path


def __parse_adjacencylist_unchecked__(f):
    '''
    read the adjacency matrix
    :param f: line stream of the file opened
    :return: the adjacency matrix
    '''
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])
    return adjlist


def __from_adjlist_unchecked__(adjlist):
    '''
    create graph form the an adjacency list
    :param adjlist: the adjacency matrix
    :return: networkx graph
    '''
    G = nx.Graph()
    G.add_edges_from(adjlist)
    return G

def load_adjacencylist(file_, undirected=False, chunksize=10000):
    '''
    multi-threaded function to read the adjacency matrix and build the graph
    :param file_: graph file
    :param undirected: is the graph undirected
    :param chunksize: how many edges for thread
    :return:
    '''

    parse_func = __parse_adjacencylist_unchecked__
    convert_func = __from_adjlist_unchecked__


    adjlist = []

    #read the matrix file
    t0 = time()
    with open(file_, 'r') as f:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            total = 0
            for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))): #execute pare_function on the adiacent list of the file in multipe process
                adjlist.extend(adj_chunk) #merge the results of different process
                total += len(adj_chunk)
    t1 = time()
    adjlist = np.asarray(adjlist)

    log.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

    t0 = time()
    G = convert_func(adjlist)
    t1 = time()

    log.debug('Converted edges to graph in {}s'.format(t1-t0))

    if undirected:
        G = G.to_undirected()

    return G


def _write_walks_to_disk(args):
    num_paths, path_length, alpha, rand, f = args
    G = __current_graph
    t_0 = time()
    with open(f, 'w') as fout:
        for walk in build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length, alpha=alpha, rand=rand):
            fout.write(u"{}\n".format(u" ".join(__vertex2str[v] for v in walk)))
    log.info("Generated new file {}, it took {} seconds".format(f, time() - t_0))
    return f

def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=cpu_count()):
    '''
    save the random walks on files so is not needed to perform the walks at each execution
    :param G: graph to walks on
    :param filebase: location where to save the final walks
    :param num_paths: number of walks to do for each node
    :param path_length: lenght of each walks
    :param alpha: restart probability for the random walks
    :param rand: generator of random numbers
    :param num_workers: number of thread used to execute the job
    :return:
    '''
    global __current_graph
    global __vertex2str
    __current_graph = G
    __vertex2str = {v:str(v) for v in G.nodes()}
    files_list = ["{}.{}".format(filebase, str(x)) for x in range(num_paths)]
    expected_size = len(G)
    args_list = []
    files = []
    log.info("file_base: {}".format(filebase))
    if num_paths <= num_workers:
        paths_per_worker = [1 for x in range(num_paths)]
    else:
        paths_per_worker = [len(list(filter(lambda z: z!= None, [y for y in x]))) for x in grouper(int(num_paths / num_workers)+1, range(1, num_paths+1))]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
            args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2**31)), file_))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)

    return files

def combine_files_iter(file_list):
    for file in file_list:
        with open(file, 'r') as f:
            for line in f:
                yield map(int, line.split())

def count_lines(f):
    if path.isfile(f):
        num_lines = sum(1 for line in open(f))
        return num_lines
    else:
        return 0

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    '''
    extract the walks form the graph used for context embeddings
    :param G: graph
    :param num_paths: how many random walks to form a sentence
    :param path_length: how long each path -> length of the sentence
    :param alpha: restart probability
    :param rand: random function
    :return:
    '''
    walks = []
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(__random_walk__(G, path_length, rand=rand, alpha=alpha, start=node))
    return np.array(walks)


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    walks = []
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield __random_walk__(G,path_length, rand=rand, alpha=alpha, start=node)


def count_textfiles(files, workers=1):
    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c

def count_words(file):
    """ Counts the word frequences in a list of sentences.

    Note:
      This is a helper function for parallel execution of `Vocabulary.from_text`
      method.
    """
    c = Counter()
    with open(file, 'r') as f:
        for l in f:
            words = [int(word) for word in l.strip().split()]
            c.update(words)
    return c

def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)

def from_numpy(x, undirected=True):
    """
    Load graph form adjmatrix
    :param x: numpy adj matrix
    :param undirected: 
    :return: 
    """
    G = nx.Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G.add_edge(i, j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G = G.to_undirected()
    return G


def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

