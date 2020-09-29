__author__ = 'ando'

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

from os.path import exists, join as path_join
from os import makedirs
import numpy as np

CAMP = 'viridis'


def _binary_commonity(G, label):
    '''
    Coloring function based on the label
    NB. label have to be binary
    :param G: Graph
    :param label: list of nodes length. for each node represent its label
    :return: list of the color for each node
    '''
    color_map = list(plt.get_cmap(CAMP)(np.linspace(0.0, 1, 4)))
    nodes_color = np.zeros((G.number_of_nodes(), 4))

    for index, node_id in enumerate(sorted(list(G.nodes()))):
        if label[index] == 1:
            nodes_color[index] = color_map[0]
        elif label[index] == 2:
            nodes_color[index] = color_map[-1]
        else:
            ValueError("Label is not binary")
    return nodes_color


def graph_plot(G,
               labels=None,
               path="./plots",
               plot_name="graph",
               save=True):
    spring_pos = nx.spring_layout(G)
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    nx.draw_networkx(G, node_color=labels, pos=spring_pos, camp=plt.get_cmap(CAMP), nodelist=sorted(G.nodes()))

    if save:
        if not exists(path):
            makedirs(path)
        plt.savefig(path + "/graph_" + plot_name + '.png')
        plt.close()
    else:
        plt.show()

    plt.clf()
    plt.close()


def node_space_plot_2d(embedding,
                       labels=None,
                       path="./plots",
                       plot_name="graph",
                       save=False,
                       grid=False):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    nodes_id = np.array(list(range(len(embedding))))
    data = np.concatenate((embedding, np.expand_dims(nodes_id, axis=1),), axis=1)

    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o', cmap=CAMP)

    for node in data:
        ax.text(node[0], node[1], '%s' % (str(int(node[2]))), size=10)

    if grid:
        x_max, x_min = -2, -4.5
        y_max, y_min = 1.5, -1.5

        x_step = (x_max - x_min) / 4.0
        y_step = (y_max - y_min) / 4.0

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        x_major_ticks = np.arange(x_min, x_max + 0.01, 2 * x_step)
        x_minor_ticks = np.arange(x_min, x_max + 0.01, x_step)

        y_major_ticks = np.arange(y_min, y_max + 0.001, 2 * y_step)
        y_minor_ticks = np.arange(y_min, y_max + 0.001, y_step)

        ax.set_xticks(x_major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)

        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)

        ax.grid(which='both')

    if save:
        if not exists(path):
            makedirs(path)
        plt.savefig(path + "/node_emb_" + plot_name + '.png')
        plt.close()
    else:
        plt.show()


def node_space_plot_2d_ellipsoid(embedding,
                                 labels=None,
                                 means=None,
                                 covariances=None,
                                 grid=False,
                                 path="./plots",
                                 plot_name=None,
                                 save=False):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    nodes_id = np.array(list(range(len(embedding))))
    data = np.concatenate((embedding, np.expand_dims(nodes_id, axis=1),), axis=1)

    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o', cmap=CAMP)

    for node in data:
        ax.text(node[0], node[1], '%s' % (str(int(node[2]))), size=10)

    ellipses = get_ellipses_artists(labels=labels, means=means, covariances=covariances)
    for ellipse in ellipses:
        ellipse.set_clip_box(ax.bbox)
        ax.add_artist(ellipse)

    if grid:
        x_max, x_min = 0.5, -1
        y_max, y_min = 2, -2

        x_step = (x_max - x_min) / 4.0
        y_step = (y_max - y_min) / 4.0

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        x_major_ticks = np.arange(x_min, x_max + 0.01, 2 * x_step)
        x_minor_ticks = np.arange(x_min, x_max + 0.01, x_step)

        y_major_ticks = np.arange(y_min, y_max + 0.001, 2 * y_step)
        y_minor_ticks = np.arange(y_min, y_max + 0.001, y_step)

        ax.set_xticks(x_major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)

        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)

        ax.grid(which='both')

    if save:
        if not exists(path):
            makedirs(path)
        plt.savefig(path + "/embeddings_" + plot_name + '.png')
        plt.close()
    else:
        plt.show()

    plt.clf()
    plt.close()


def get_ellipses_artists(labels=None,
                         means=None,
                         covariances=None):
    artists = []
    if (means is not None) and (covariances is not None):
        for i, (mean, covar) in enumerate(zip(means, covariances)):
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(labels == i):
                continue

            # higher dim -> 2D
            mean = mean[:2]
            covar = covar[:2,:2]

            # computations for showing ellipses
            v, w = np.linalg.eigh(2.5 * covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])

            transparency = 0.45

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ellipse = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, fill=False, linewidth=2.)
            # ellipse.set_clip_box(ax.bbox)
            ellipse.set_alpha(transparency)

            artists.append(ellipse)

    return artists


def animate_step(ax, model, i=None, i_com=None, converged=False, max_nodes=999999, show_node_ids=True):
    # extract parameters
    # nodes
    nodes = model.node_embedding
    labels = model.classify_nodes()
    # down sampling
    if max_nodes < len(nodes):
        nodes_i = np.random.choice(len(nodes), max_nodes, replace=False)
        nodes = nodes[nodes_i]
        labels = labels[nodes_i]
    # communities
    means = model.centroid
    covars = model.covariance_mat

    # animation
    # counter
    counter = ax.text(0.05, 0.95, f"{i}.{i_com}{' converged' if converged else ''}", fontsize=16, horizontalalignment='left',
                      verticalalignment='top', transform=ax.transAxes)
    # nodes
    nodes_scatter = ax.scatter(nodes[:, 0], nodes[:, 1], 20, c=labels, marker="o")
    nodes_ids = []
    if show_node_ids:
        for (i_node, node) in enumerate(nodes):
            nodes_ids.append(ax.text(node[0], node[1], str(i_node), size=12))
    # communities
    ellipses = get_ellipses_artists(labels=labels, means=means, covariances=covars)
    for ellipse in ellipses:
        ellipse.set_clip_box(ax.bbox)
        ax.add_artist(ellipse)

    # return artists
    return ellipses + nodes_ids + [nodes_scatter, counter]


def bar_plot_bgmm_weights(weights,
                          path="./plots",
                          plot_name=None,
                          save=False):
    plt.bar(np.arange(len(weights)), weights)

    if save:
        if not exists(path):
            makedirs(path)
        plt.savefig(path + "/weights_" + plot_name + '.png')
        plt.close()
    else:
        plt.show()
