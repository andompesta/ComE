__author__ = 'ando'

import os
import random
from multiprocessing import cpu_count
import logging as log
import joblib

import numpy as np
import psutil
from math import floor

from sklearn import metrics

from ADSCModel.model import Model
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.node_embeddings import Node2Vec
from ADSCModel.community_embeddings import Community2Vec
from utils.IO_utils import save_embedding, load_ground_true
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import timeit
import networkx as nx

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

    number_walks = 10  # γ: number of walks for each node
    walk_length = 80  # l: length of each walk
    representation_size = 2  # size of the embedding
    num_workers = 10  # number of thread
    num_iter = 3  # number of overall iteration
    reg_covar = 0.00001  # regularization coefficient to ensure positive covar
    input_file = 'karate_club'  # name of the input file
    output_file = 'karate_club'  # name of the output file
    batch_size = 50
    window_size = 10  # ζ: windows size used to compute the context embedding
    negative = 5  # m: number of negative sample
    lr = 0.025  # learning rate
    alpha_betas = [(0.1, 0.1)]  # Trade-off parameter for context/community embedding
    down_sampling = 0.0

    come_model_type = "BGMM"  # type of the Community Embedding model: GMM/BGMM
    weight_concentration_prior = 1e-5  # dirichlet concentration of each BGMM component to (de)activate components

    ks = [5]  # number of communities to initialize the GMM/BGMM with
    walks_filebase = os.path.join('data', output_file)  # where read/write the sampled path

    # CONSTRUCT THE GRAPH
    #G = graph_utils.load_matfile(os.path.join('./data', input_file, input_file + '.mat'), undirected=True)
    G = nx.karate_club_graph()  # DEBUG run on karate club graph

    # Sampling the random walks for context
    log.info("sampling the paths")
    walk_files = graph_utils.write_walks_to_disk(G, os.path.join(walks_filebase, "{}.walks".format(output_file)),
                                                 num_paths=number_walks,
                                                 path_length=walk_length,
                                                 alpha=0,
                                                 rand=random.Random(0),
                                                 num_workers=num_workers)

    vertex_counts = graph_utils.count_textfiles(walk_files, num_workers)
    model = Model(vertex_counts,
                  size=representation_size,
                  down_sampling=down_sampling,
                  table_size=100000000)

    # Learning algorithm
    node_learner = Node2Vec(workers=num_workers, negative=negative, lr=lr)
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)
    com_learner = Community2Vec(lr=lr, model_type=come_model_type)

    context_total_path = G.number_of_nodes() * number_walks * walk_length
    edges = np.array(G.edges())
    log.debug("context_total_path: %d" % context_total_path)
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
                       alpha=1,
                       chunksize=batch_size)
    #
    model.save("{}_pre-training".format(output_file))

    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    iter_node = floor(context_total_path / G.number_of_edges())
    iter_com = floor(context_total_path / G.number_of_edges())
    # iter_com = 1
    # alpha, beta = alpha_betas

    for i in range(num_iter):
        for alpha, beta in alpha_betas:
            for k in ks:
                log.info('\n_______________________________________\n')
                log.info(f'\t\tITER-{i}\n')
                model = model.load_model(f"{output_file}_pre-training")
                model.reset_communities_weights(k)
                log.info(f'using alpha:{alpha}\tbeta:{beta}\titer_com:{iter_com}\titer_node: {iter_node}')
                start_time = timeit.default_timer()

                com_learner.fit(model,
                                weight_concentration_prior=weight_concentration_prior,
                                reg_covar=reg_covar,
                                n_init=10)
                node_learner.train(model,
                                   edges=edges,
                                   iter=iter_node,
                                   chunksize=batch_size)

                com_learner.train(G.nodes(), model, beta, chunksize=batch_size, iter=iter_com)

                cont_learner.train(model,
                                   paths=graph_utils.combine_files_iter(walk_files),
                                   total_nodes=context_total_path,
                                   alpha=alpha,
                                   chunksize=batch_size)

                log.info('time: %.2fs' % (timeit.default_timer() - start_time))
                # log.info(model.centroid)
                save_embedding(model.node_embedding, model.vocab,
                               file_name=f"{output_file}_alpha-{alpha}_beta-{beta}_ws-{window_size}_neg-{negative}_lr-{lr}_icom-{iter_com}_ind-{iter_node}_k-{model.k}_ds-{down_sampling}")

                # DEBUG plot after each iter
                plot_utils.node_space_plot_2d_ellipsoid(model.node_embedding,
                                                        labels=model.classify_nodes(),
                                                        means=com_learner.g_mixture.means_,
                                                        covariances=com_learner.g_mixture.covariances_,
                                                        plot_name=f"k{k}_i{i}",
                                                        save=True)

    # ### print model
    node_classification = model.classify_nodes()
    print("model:\n",
          "  model.node_embedding: ", model.node_embedding, "\n",
          "  model.context_embedding: ", model.context_embedding, "\n",
          "  model.centroid: ", model.centroid, "\n",
          "  model.covariance_mat: ", model.covariance_mat, "\n",
          "  model.inv_covariance_mat: ", model.inv_covariance_mat, "\n",
          "  model.pi: ", model.pi, "\n",
          "=>node_classification: ", node_classification, "\n", )

    # ### write predictions to labels_pred.txt
    # save com_learner.g_mixture to file
    joblib.dump(com_learner.g_mixture, './data/g_mixture.joblib')
    # using predictions from com_learner.g_mixture with node_embeddings
    np.savetxt('./data/labels_pred.txt', model.classify_nodes())

    # ### NMI
    labels_true, _ = load_ground_true(path="data/"+input_file, file_name=input_file)
    print("labels_true: ", labels_true)
    if labels_true is not None:
        nmi = metrics.normalized_mutual_info_score(labels_true, node_classification)
        print("===NMI=== ", nmi)
    else:
        print("===NMI=== could not be computed")

    # ### plotting
    plot_name = str(ks[0])

    if representation_size == 2:
        # graph_plot
        plot_utils.graph_plot(G, labels=node_classification, plot_name=plot_name, save=True)

        # node_space_plot_2D
        plot_utils.node_space_plot_2d(model.node_embedding, labels=node_classification, plot_name=plot_name, save=True)

        # node_space_plot_2d_ellipsoid
        plot_utils.node_space_plot_2d_ellipsoid(model.node_embedding,
                                                labels=node_classification,
                                                means=com_learner.g_mixture.means_,
                                                covariances=com_learner.g_mixture.covariances_,
                                                plot_name=plot_name,
                                                save=True)

    # bar_plot_bgmm_pi
    plot_utils.bar_plot_bgmm_weights(com_learner.g_mixture.weights_, plot_name=plot_name, save=True)
