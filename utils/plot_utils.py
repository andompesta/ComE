__author__ = 'ando'

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from os.path import exists, join as path_join
from os import makedirs
import numpy as np
import itertools
import pickle

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
               show=True):
    spring_pos = nx.spring_layout(G)
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    nx.draw_networkx(G, node_color=labels, pos=spring_pos, camp=plt.get_cmap(CAMP), nodelist=sorted(G.nodes()))

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


def node_space_plot_2d(embedding,
                       labels=None,
                       path="graph",
                       graph_name='graph',
                       save=False,
                       grid=False):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    nodes_id = np.array(list(range(1, len(embedding) + 1)))
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
        plt.savefig(path + graph_name + '_prj_2d' + '.png')
        plt.close()
    else:
        plt.show()


def node_space_plot_2d_ellipsoid(embedding,
                                 labels=None,
                                 means=None,
                                 covariances=None,
                                 grid=False,
                                 path='./graph',
                                 plot_name=None,
                                 show=True):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    nodes_id = np.array(list(range(1, len(embedding) + 1)))
    data = np.concatenate((embedding, np.expand_dims(nodes_id, axis=1),), axis=1)

    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o', cmap=CAMP)

    for node in data:
        ax.text(node[0], node[1], '%s' % (str(int(node[2]))), size=10)

    if (means is not None) and (covariances is not None):
        for i, (mean, covar) in enumerate(zip(means, covariances)):
            v, w = np.linalg.eigh(2.5 * covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            transparency = 0.45

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, fill=False, linewidth=2.)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(transparency)
            ax.add_artist(ell)

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

    if plot_name:
        if not exists(path):
            makedirs(path)
        plt.savefig(path_join(path, plot_name + '.png'))

    if show:
        plt.show()

    plt.clf()
    plt.close()
