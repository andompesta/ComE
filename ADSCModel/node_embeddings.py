__author__ = 'ando'

import logging as log
log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

import time
import threading
from queue import Queue
import numpy as np
from utils.embedding import chunkize_serial, RepeatCorpusNTimes, prepare_sentences
from scipy.special import expit as sigmoid

from utils.training_sdg_inner import train_o1, loss_o1, FAST_VERSION
log.info("imported cython version: {}".format(FAST_VERSION))



class Node2Vec(object):
    def __init__(self, lr=0.2, workers=1, negative=0):

        self.workers = workers
        self.lr = float(lr)
        self.min_lr = 0.0001
        self.negative = negative
        self.window_size = 1

    def loss(self, model, edges):
        ret_loss = 0
        for edge in prepare_sentences(model, edges):
            assert len(edge) == 2, "edges have to be done by 2 nodes :{}".format(edge)
            edge_loss = np.log(
                sigmoid(np.dot(model.node_embedding[edge[1].index], model.node_embedding[edge[0].index].T)))
            assert edge_loss <= 0,"malformed loss"
            ret_loss -= edge_loss
        return ret_loss
    #
    # def loss(self, model, edges):
    #     loss = 0.0
    #     num_nodes = 0
    #
    #     for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, edges), 250)):
    #         batch_loss = np.zeros(1, dtype=np.float32)
    #         batch_work = np.zeros(model.layer1_size, dtype=np.float32)
    #
    #
    #         batch_node = sum([loss_o1(model.node_embedding, edge, self.negative, model.table,
    #                              py_size=model.layer1_size, py_loss=batch_loss, py_work=batch_work) for edge in job if edge is not None])
    #         num_nodes += batch_node
    #         loss += batch_loss[0]
    #         # log.info("loss: {}\tnodes: {}".format(loss, num_nodes))
    #
    #     log.info(num_nodes)
    #     log.info(loss)
    #     return loss



    def train(self, model, edges, chunksize=150, iter=1):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        assert model.node_embedding.dtype == np.float32

        log.info("O1 training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        edges = RepeatCorpusNTimes(edges, iter)
        total_node = edges.corpus.shape[0] * edges.corpus.shape[1] * edges.n
        log.debug('total edges: %d' % total_node)
        start, next_report, node_count = time.time(), [5.0], [0]

        #int(sum(v.count * v.sample_probability for v in self.vocab.values()))
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()


        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            py_work = np.zeros(model.layer1_size, dtype=np.float32)

            while True:
                job = jobs.get(block=True)
                if job is None:  # data finished, exit
                    jobs.task_done()
                    # print('thread %s break' % threading.current_thread().name)
                    break

                lr = max(self.min_lr, self.lr * (1 - 1.0 * node_count[0]/total_node))
                job_words = sum(train_o1(model.node_embedding, edge, lr, self.negative, model.table,
                                         py_size=model.layer1_size, py_work=py_work) for edge in job if edge is not None)
                jobs.task_done()
                lock.acquire(timeout=30)
                try:
                    node_count[0] += job_words

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        log.info("PROGRESS: at %.2f%% \tnode_computed %d\talpha %.05f\t %.0f nodes/s" %
                                    (100.0 * node_count[0] / total_node, node_count[0], lr, node_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 5.0  # don't flood the log, wait at least a second between progress reports
                finally:
                    lock.release()



        workers = [threading.Thread(target=worker_train, name='thread_'+str(i)) for i in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()


        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, edges), chunksize)):
            jobs.put(job)


        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        log.info("training on %i words took %.1fs, %.0f words/s" %
                    (node_count[0], elapsed, node_count[0]/ elapsed if elapsed else 0.0))
