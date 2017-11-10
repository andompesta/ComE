__author__ = 'ando'
import os
import random
from multiprocessing import cpu_count
import logging as log


import numpy as np
import psutil
from math import floor
from ADSCModel.model import Model
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.node_embeddings import Node2Vec
from ADSCModel.community_embeddings import Community2Vec
import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import timeit

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)




p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

if __name__ == "__main__":

    #Reading the input parameters form the configuration files
    number_walks = 10                       # number of walks for each node
    walk_length = 20                        # length of each walk
    representation_size = 2               # size of the embedding
    num_workers = 10                        # number of thread
    num_iter = 1                            # number of overall iteration
    reg_covar = 0.001                    # regularization coefficient to ensure positive covar
    input_file = 'Karate'                # name of the input file
    output_file = 'Karate'               # name of the output file
    batch_size = 10
    window_size = 3    # windows size used to compute the context embedding
    negative = 5        # number of negative sample
    lr = 0.025           # learning rate

    alpha_betas = [(1., 0.01)]
    down_sampling = 0.0

    k = 2
    walks_filebase = os.path.join('data', output_file)            # where read/write the sampled path



    #CONSTRUCT THE GRAPH
    G = graph_utils.load_adjacencylist(os.path.join('./data', input_file, input_file + '.adjlist'), undirected=True)
    # Sampling the random walks for context
    log.info("sampling the paths")
    walk_files = graph_utils.write_walks_to_disk(G,
                                                 os.path.join(walks_filebase, "{}.walks".format(output_file)),
                                                 num_paths=number_walks,
                                                 path_length=walk_length,
                                                 alpha=0,
                                                 rand=random.Random(0),
                                                 num_workers=num_workers)

    vertex_counts = graph_utils.count_textfiles(walk_files, num_workers)
    model = Model(vertex_counts,
                  size=representation_size,
                  down_sampling=down_sampling,
                  table_size=5000000,
                  input_file=os.path.join(input_file, input_file),
                  path_labels="./data")


    #Learning algorithm
    node_learner = Node2Vec(workers=num_workers, negative=negative, lr=lr)
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)
    com_learner = Community2Vec(lr=lr)


    context_total_path = G.number_of_nodes() * number_walks * walk_length
    edges = np.array(G.edges())
    log.debug("context_total_path: %d" % (context_total_path))
    log.debug('node total edges: %d' % G.number_of_edges())

    log.info('\n_______________________________________')
    log.info('\t\tPRE-TRAINING\n')
    ###########################
    #   PRE-TRAINING          #
    ###########################
    node_learner.train(model,
                       edges=edges,
                       iter=1,
                       chunksize=batch_size)

    cont_learner.train(model,
                       paths=graph_utils.combine_files_iter(walk_files),
                       total_nodes=context_total_path,
                       alpha=1.0,
                       chunksize=batch_size)
    model.save("{}_pre-training".format(output_file))

    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    iter_node = 1
    iter_com = 1
    # alpha, beta = alpha_betas

    for it in range(1):
        for alpha, beta in alpha_betas:
            log.info('\n_______________________________________\n')
            log.info('\t\tITER-{}\n'.format(it))
            log.info('using alpha:{}\tbeta:{}\titer_com:{}\titer_node: {}'.format(alpha, beta, iter_com, iter_node))

            start_time = timeit.default_timer()

            node_learner.train(model,
                               edges=edges,
                               iter=iter_node,
                               chunksize=batch_size)



            cont_learner.train(model,
                               paths=graph_utils.combine_files_iter(walk_files),
                               total_nodes=context_total_path,
                               alpha=alpha,
                               chunksize=batch_size)
            com_learner.fit(model, reg_covar=reg_covar, n_init=10)
            com_learner.train(G.nodes(), model, beta, chunksize=batch_size, iter=iter_com)

            log.info('time: %.2fs' % (timeit.default_timer() - start_time))
            node_color = plot_utils.graph_plot(G,
                                               graph_name=input_file,
                                               show=False)

            plot_utils.node_space_plot_2D_elipsoid(model.node_embedding, node_color,
                                                   means=model.centroid,
                                                   covariances=model.covariance_mat,
                                                   show=True)


            # io_utils.save_embedding(model.node_embedding, model.vocab,
            #                         file_name="{}_alpha-{}_beta-".format(output_file,
            #                                                              alpha,
            #                                                              beta))

