__author__ = 'ando'

import sklearn.mixture as mixture
import numpy as np
from utils.embedding import chunkize_serial

from scipy.stats import multivariate_normal
import logging as log

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)


class Community2Vec(object):
    """
    Class that train the community embedding
    """

    def __init__(self, lr, model_type="GMM"):
        """
        :param lr: learning rate
        :param model_type: GMM: Gaussian Mixture Model (default) or BGMM: Bayesian Gaussian Mixture Model (BGMM)
        """
        self.lr = lr
        self.model_type = model_type
        self.g_mixture = None

    def reset_mixture(self, model, reg_covar=0, n_init=10, max_iter=None, random_state=None,
                      weight_concentration_prior=None):
        """
        Fit the GMM/BGMM model with the current node embedding and save the result in the model
        :param model: model injected to add the mixture parameters
        :param reg_covar: non-negative regularization added to the diagonal of covariance
        :param n_init: number of initializations to perform
        :param max_iter: maximum number of iterations to run
        :param random_state: random state to use for reproducibility
        :param weight_concentration_prior: dirichlet concentration of each component (gamma). default: 1/n_components
        """
        self.g_mixture = self.get_mixture(model.k, reg_covar, n_init, max_iter, random_state,
                                          weight_concentration_prior)
        # self._update_model(model) TODO: how to get mean and covar for iter=0 (init values)?

    def fit(self, model):
        """
        Fit the GMM/BGMM model with the current node embedding and save the result in the model
        :param model: model injected to add the mixture parameters
        """

        # log.info("Fitting: {} communities".format(model.k))
        self.g_mixture.fit(model.node_embedding)
        self._update_model(model)

    def _update_model(self, model):
        model.centroid = self.g_mixture.means_.astype(np.float32)
        model.covariance_mat = self.g_mixture.covariances_.astype(np.float32)
        model.inv_covariance_mat = self.g_mixture.precisions_.astype(np.float32)
        model.pi = self.g_mixture.predict_proba(model.node_embedding).astype(np.float32)

    def get_mixture(self, k, reg_covar=0, n_init=10, max_iter=420, random_state=None, weight_concentration_prior=None):
        def get_gmm():
            return mixture.GaussianMixture(n_components=k,
                                           reg_covar=reg_covar,
                                           covariance_type='full',
                                           n_init=n_init,
                                           max_iter=max_iter,
                                           random_state=random_state,
                                           init_params='random',
                                           )

        def get_bgmm():
            return mixture.BayesianGaussianMixture(n_components=k,
                                                   weight_concentration_prior=weight_concentration_prior,
                                                   reg_covar=reg_covar,
                                                   covariance_type='full',
                                                   n_init=n_init,
                                                   max_iter=max_iter,
                                                   random_state=random_state,
                                                   init_params='random',
                                                   )

        if self.model_type == "BGMM":
            return get_bgmm()
        elif self.model_type == "GMM":
            return get_gmm()
        else:
            log.warning(f"Unknown ComE model type {self.model_type}. Defaulting to GMM.")
            return get_gmm()

    def loss(self, nodes, model, beta, chunksize=150):
        """
        Forward function used to compute o3 loss
        :param nodes:
        :param model: model containing all the shared data
        :param beta: trade off param
        :param chunksize:
        """
        ret_loss = 0
        for node_index in chunkize_serial(map(lambda x: model.vocab[x].index, nodes), chunksize):
            input = model.node_embedding[node_index]

            batch_loss = np.zeros(len(node_index), dtype=np.float32)
            for com in range(model.k):
                rd = multivariate_normal(model.centroid[com], model.covariance_mat[com])
                # check if can be done as matrix operation
                batch_loss += rd.logpdf(input).astype(np.float32) * model.pi[node_index, com]

            ret_loss = abs(batch_loss.sum())

        return ret_loss * (beta / model.k)

    def train(self, nodes, model, beta, chunksize=150, iter=1):
        """
        :param nodes:
        :param model: model containing all the shared data
        :param beta: trade off param
        :param chunksize:
        :param iter:
        """
        for _ in range(iter):
            grad_input = np.zeros(model.node_embedding.shape).astype(np.float32)
            for node_index in chunkize_serial(map(lambda node: model.vocab[node].index,
                                                  filter(lambda node: node in model.vocab and (
                                                          model.vocab[node].sample_probability >= 1.0 or model.vocab[
                                                      node].sample_probability >= np.random.random_sample()), nodes)),
                                              chunksize):
                input = model.node_embedding[node_index]
                batch_grad_input = np.zeros(input.shape).astype(np.float32)

                for com in range(model.k):
                    diff = np.expand_dims(input - model.centroid[com], axis=-1)
                    m = model.pi[node_index, com].reshape(len(node_index), 1, 1) * (model.inv_covariance_mat[com])

                    batch_grad_input += np.squeeze(np.matmul(m, diff), axis=-1)
                grad_input[node_index] += batch_grad_input

            grad_input *= (beta / model.k)

            model.node_embedding -= (grad_input.clip(min=-0.25, max=0.25)) * self.lr

    @property
    def converged(self):
        if self.g_mixture:
            return self.g_mixture.converged_
        else:
            return False

    @property
    def n_iter(self):
        if self.g_mixture:
            return self.g_mixture.n_iter_
        else:
            return '?'
