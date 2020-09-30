__author__ = 'ando'

import os
import random
from multiprocessing import cpu_count
import logging as log
from itertools import product
import joblib

import numpy as np
import psutil
from math import floor

from sklearn import metrics
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from ADSCModel.model import Model
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.node_embeddings import Node2Vec
from ADSCModel.community_embeddings import Community2Vec
from utils.IO_utils import save_embedding, load_ground_true
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import timeit
import networkx as nx  # DEBUG

import seaborn  # fancy matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

# set random number generator seed
random_state = 2020
np.random.seed(random_state)

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

if __name__ == "__main__":

    verbose = True
    should_animate = True
    should_plot_steps = True
    should_plot = True
    com_iter_step = 100

    number_walks = 10  # γ: number of walks for each node
    walk_length = 80  # l: length of each walk
    representation_size = 4  # size of the embedding
    num_workers = 10  # number of thread
    num_iter = 3  # number of overall iteration
    com_n_init = 10  # number of inits for community embedding (default: 10)
    reg_covar = 0.00001  # regularization coefficient to ensure positive covar
    input_file = 'facebook'  # name of the input file
    output_file = input_file  # name of the output file
    batch_size = 50
    window_size = 10  # ζ: windows size used to compute the context embedding
    negative = 5  # m: number of negative sample
    lr = 0.025  # learning rate
    alpha_betas = [(0.1, 0.1)]  # Trade-off parameter for context/community embedding
    down_sampling = 0.0

    come_model_type = "BGMM"  # type of the Community Embedding model: GMM/BGMM
    weight_concentration_prior = 1e-5  # dirichlet concentration of each BGMM component to (de)activate components

    ks = [10]  # number of communities to initialize the GMM/BGMM with
    walks_filebase = os.path.join('data', output_file)  # where read/write the sampled path

    # CONSTRUCT THE GRAPH

    # load from matfile
    # G = graph_utils.load_matfile(os.path.join('./data', input_file, input_file + '.mat'), undirected=True)
    # load karate club directly
    # G = nx.karate_club_graph()  # DEBUG run on karate club graph, make sure to mkdir ./data/karate_club

    # load from edgelist csv
    G = graph_utils.load_edgelist(os.path.join('./data', input_file, input_file + '.csv'), source="u", target="v")

    # DEBUG remove some edges for karate_club
    '''
    print("PRE NUM_OF_EDGES: ", G.number_of_edges())
    G.remove_edge(33, 23)
    G.remove_edge(0, 1)
    G.remove_edge(32, 30)
    print("POST NUM_OF_EDGES: ", G.number_of_edges())
    '''

    print("G.number_of_nodes: ", G.number_of_nodes())
    print("G.number_of_edges: ", G.number_of_edges())

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
    model.save(f"{output_file}_pre-training")

    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    iter_node = floor(context_total_path / G.number_of_edges())
    iter_com = floor(context_total_path / G.number_of_edges())
    log.info(f'using iter_com:{iter_com}\titer_node: {iter_node}')

    anim_fig = plt.figure(figsize=(8, 8))
    anim_ax = anim_fig.add_subplot(111)
    anim_artists = []

    for (alpha, beta), k in product(alpha_betas, ks):
        log.info('\n_______________________________________\n')
        log.info(f'TRAINING \t\talpha:{alpha}\tbeta:{beta}\tk:{k}')
        model = model.load_model(f"{output_file}_pre-training")
        model.reset_communities_weights(k)

        for i in range(num_iter):
            log.info(f'\t\tITER-{i}\n')
            com_max_iter = 0
            start_time = timeit.default_timer()

            while not com_learner.converged or com_max_iter == 0:
                params_anim = {}
                com_max_iter += com_iter_step if should_animate else 100
                if should_animate:
                    if verbose:
                        log.info(f"->com_max_iter={com_max_iter}")
                    params_anim['max_iter'] = com_max_iter

                com_learner.reset_mixture(model,
                                          reg_covar=reg_covar,
                                          n_init=com_n_init,
                                          random_state=random_state,
                                          weight_concentration_prior=weight_concentration_prior,
                                          **params_anim)

                with ignore_warnings(category=ConvergenceWarning):
                    com_learner.fit(model)

                def animate_model():
                    if should_animate:
                        artists_step = plot_utils.animate_step(anim_ax,
                                                               model,
                                                               i=i,
                                                               i_com=com_learner.n_iter,
                                                               converged=com_learner.converged,
                                                               show_node_ids=False,
                                                               show_communities=False)
                        anim_artists.append(artists_step)


                # community converged?
                if not com_learner.converged:
                    log.info(f'iter {i}.{com_learner.n_iter} did not converge.')
                else:
                    log.info(f'iter {i}.{com_learner.n_iter} converged!')

                if should_animate:
                    animate_model()

                # DEBUG plot after each community iteration
                if not should_animate and should_plot_steps:
                    plot_utils.node_space_plot_2d_ellipsoid(model.node_embedding,
                                                            labels=model.classify_nodes(),
                                                            means=com_learner.g_mixture.means_,
                                                            covariances=com_learner.g_mixture.covariances_,
                                                            plot_name=f"{come_model_type}_k{k}_i{i}_{com_max_iter:03}",
                                                            path=f"./plots/{output_file}",
                                                            save=True)

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

            if verbose:
                log.info('time: %.2fs' % (timeit.default_timer() - start_time))
            save_embedding(model.node_embedding, model.vocab,
                           path=f"data/{output_file}",
                           file_name=f"{output_file}_alpha-{alpha}_beta-{beta}_ws-{window_size}_neg-{negative}_lr-{lr}_icom-{iter_com}_ind-{iter_node}_ds-{down_sampling}_d-{representation_size}_type-{come_model_type}_k-{model.k}")

            # DEBUG plot after each ComE iteration
            if not should_animate and should_plot_steps:
                plot_utils.node_space_plot_2d_ellipsoid(model.node_embedding,
                                                        labels=model.classify_nodes(),
                                                        means=com_learner.g_mixture.means_,
                                                        covariances=com_learner.g_mixture.covariances_,
                                                        plot_name=f"{come_model_type}_d{representation_size}_k{k}_i{i}",
                                                        path=f"./plots/{output_file}",
                                                        save=True)

        # compute classifications
        node_classification = model.classify_nodes()

        # ### print model
        if verbose:
            print("model:\n",
                  "  model.node_embedding: ", model.node_embedding, "\n",
                  "  model.context_embedding: ", model.context_embedding, "\n",
                  "  model.centroid: ", model.centroid, "\n",
                  "  model.covariance_mat: ", model.covariance_mat, "\n",
                  "  model.inv_covariance_mat: ", model.inv_covariance_mat, "\n",
                  "  model.pi: ", model.pi, "\n",
                  "=>node_classification: ", node_classification, "\n", )

        # ### Animation
        if should_animate:
            anim = ArtistAnimation(anim_fig, anim_artists, interval=500, blit=True, repeat=False)
            # anim.to_html5_video()
            # export animation as gif:
            # you may need to install "imagemagick" (ex.: brew install imagemagick)
            anim.save(f"./plots/{output_file}/animation_{come_model_type}_d{representation_size}_k{k}.gif", writer='imagemagick')

        # ### write predictions to labels_pred.txt
        # save com_learner.g_mixture to file
        joblib.dump(com_learner.g_mixture, f'./model/g_mixture_{output_file}_{come_model_type}_d{representation_size}_k{k}.joblib')
        # using predictions from com_learner.g_mixture with node_embeddings
        np.savetxt(f'./data/{output_file}/labels_pred_{come_model_type}_d{representation_size}_k{k}.txt', model.classify_nodes())

        # ### NMI
        labels_true, _ = load_ground_true(path="data/" + input_file, file_name=input_file)
        print("labels_true: ", labels_true)
        if labels_true is not None:
            nmi = metrics.normalized_mutual_info_score(labels_true, node_classification)
            print(f"===NMI=== for type={come_model_type} with d={representation_size} and K={k}: ", nmi)
        else:
            print(f"===NMI=== for type={come_model_type} with d={representation_size} and K={k} could not be computed")

        # ### plotting
        plot_name = f"{come_model_type}_d{representation_size}_k{k}"
        if should_plot:
            # graph_plot
            plot_utils.graph_plot(G,
                                  labels=node_classification,
                                  plot_name=plot_name,
                                  path=f"./plots/{output_file}",
                                  save=True,
                                  show_labels=False)
            # node_space_plot_2D
            plot_utils.node_space_plot_2d(model.node_embedding,
                                          labels=node_classification,
                                          plot_name=plot_name,
                                          path=f"./plots/{output_file}",
                                          save=True)
            # node_space_plot_2d_ellipsoid
            plot_utils.node_space_plot_2d_ellipsoid(model.node_embedding,
                                                    labels=node_classification,
                                                    means=com_learner.g_mixture.means_,
                                                    covariances=com_learner.g_mixture.covariances_,
                                                    plot_name=plot_name,
                                                    path=f"./plots/{output_file}",
                                                    save=True)
            # bar_plot_bgmm_pi
            plot_utils.bar_plot_bgmm_weights(com_learner.g_mixture.weights_,
                                             plot_name=plot_name,
                                             path=f"./plots/{output_file}",
                                             save=True)
