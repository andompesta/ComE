import cython
import numpy as np
cimport numpy as np
from cpython cimport PyCapsule_GetPointer
import sys

from libc.math cimport exp, isnan, log
from libc.string cimport memset
import scipy.linalg.blas as fblas

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)


REAL = np.float32
ctypedef np.float32_t REAL_t

DEF MAX_SENTENCE_LEN = 10000

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil
ctypedef void (*sgemm_ptr) (char *transA, char *transB, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc) nogil
ctypedef void (*dgemm_ptr) (char *transA, char *transB, int *m, int *n, int *k, float *alpha, double *a, int *lda, double *b, int *ldb, float *beta, double *c, int *ldc) nogil

ctypedef unsigned long long (*fast_o1_ptr) (
    const int negative,
    np.uint32_t *table,
    unsigned long long table_len,
    REAL_t *node_embedding,
    REAL_t *negative_embedding,
    const int size,
    const np.uint32_t word_index,
    const np.uint32_t word2_index,
    const REAL_t lr,
    REAL_t *work,
    unsigned long long next_random
    ) nogil




ctypedef unsigned long long (*fast_o2_ptr) (
    const int negative,
    np.uint32_t *table,
    unsigned long long table_len,
    REAL_t *node_embedding,
    REAL_t *context_embedding,
    const int size,
    const np.uint32_t word_index,
    const np.uint32_t word2_index,
    const REAL_t lr,
    const REAL_t _lambda,
    REAL_t *work,
    unsigned long long next_random
    ) nogil


cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x

cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)

cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

cdef sgemm_ptr sgemm=<sgemm_ptr>PyCObject_AsVoidPtr(fblas.sgemm._cpointer) # float x = alpha * (A B) + beta * (x)
cdef dgemm_ptr dgemm=<dgemm_ptr>PyCObject_AsVoidPtr(fblas.sgemm._cpointer) # double x = alpha * (A B) + beta * (x)

cdef fast_o1_ptr fast_o1
cdef fast_o2_ptr fast_o2

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t NONEF = <REAL_t>-1.0

cdef REAL_t DECF = <REAL_t>0.5
cdef REAL_t NDECF = <REAL_t>-0.5


cdef unsigned long long fast0_o2 (
    const int negative,
    np.uint32_t *table,
    unsigned long long table_len,
    REAL_t *node_embedding,
    REAL_t *context_embedding,
    const int size,
    const np.uint32_t word_index,
    const np.uint32_t word2_index,
    const REAL_t lr,
    const REAL_t _lambda,
    REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>dsdot(&size, &node_embedding[row1], &ONE, &context_embedding[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * (_lambda)
        g = g * lr
        saxpy(&size, &g, &context_embedding[row2], &ONE, work, &ONE) # work += g * negative_embeddings
        saxpy(&size, &g, &node_embedding[row1], &ONE, &context_embedding[row2], &ONE) # negative_embeddings += g * node_embedding

    saxpy(&size, &ONEF, work, &ONE, &node_embedding[row1], &ONE)  #node_embedding += work

    return next_random



cdef unsigned long long fast1_o2 (
    const int negative,
    np.uint32_t *table,
    unsigned long long table_len,
    REAL_t *node_embedding,
    REAL_t *context_embedding,
    const int size,
    const np.uint32_t word_index,
    const np.uint32_t word2_index,
    const REAL_t lr,
    const REAL_t _lambda,
    REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):

        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>sdot(&size, &node_embedding[row1], &ONE, &context_embedding[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * (_lambda)
        g = g * lr
        saxpy(&size, &g, &context_embedding[row2], &ONE, work, &ONE)
        saxpy(&size, &g, &node_embedding[row1], &ONE, &context_embedding[row2], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &node_embedding[row1], &ONE)

    return next_random


cdef unsigned long long fast0_o1 (
    const int negative,
    np.uint32_t *table,
    unsigned long long table_len,
    REAL_t *node_embedding,
    REAL_t *negative_embedding,
    const int size,
    const np.uint32_t word_index,
    const np.uint32_t word2_index,
    const REAL_t lr,
    REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>dsdot(&size, &node_embedding[row1], &ONE, &negative_embedding[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f)
        g = g * lr

        saxpy(&size, &g, &negative_embedding[row2], &ONE, work, &ONE) # work += g * negative_embeddings

    saxpy(&size, &ONEF, work, &ONE, &node_embedding[row1], &ONE)  #node_embedding += _lambda * work

    return next_random


cdef unsigned long long fast1_o1 (
    const int negative,
    np.uint32_t *table,
    unsigned long long table_len,
    REAL_t *node_embedding,
    REAL_t *negative_embedding,
    const int size,
    const np.uint32_t word_index,
    const np.uint32_t word2_index,
    const REAL_t lr,
    REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = <REAL_t>sdot(&size, &node_embedding[row1], &ONE, &negative_embedding[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f)
        g = g * lr

        saxpy(&size, &g, &negative_embedding[row2], &ONE, work, &ONE) # work += g * negative_embeddings

    saxpy(&size, &ONEF, work, &ONE, &node_embedding[row1], &ONE)  #node_embedding += _lambda * work

    return next_random


cdef unsigned long long loss_o1_o2 (
    const int negative,
    np.uint32_t *table,
    unsigned long long table_len,
    REAL_t *node_embedding,
    REAL_t *negative_embedding,
    const int size,
    const np.uint32_t word_index,
    const np.uint32_t word2_index,
    const REAL_t _lambda,
    REAL_t *loss,
    REAL_t *work,
    unsigned long long next_random) nogil:

    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d


    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = NONEF

        row2 = target_index * size

        scopy(&size, &negative_embedding[row2], &ONE, work, &ONE) # work = &negative_embedding[row2]
        sscal(&size, &label, work, &ONE) # work = work * label


        f = <REAL_t>sdot(&size, &node_embedding[row1], &ONE, work, &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = _lambda * log(f)

        saxpy(&ONE, &g, &NONEF, &ONE, loss, &ONE) # loss += g * 1
    return next_random

def loss_o1(py_node_embedding, py_edge, py_negative, py_table,
             py_size=None, py_work=None, py_loss=None):

    cdef REAL_t *node_embedding = <REAL_t *>(np.PyArray_DATA(py_node_embedding))
    cdef REAL_t *work = <REAL_t *>(np.PyArray_DATA(py_work))
    cdef REAL_t *loss = <REAL_t *>(np.PyArray_DATA(py_loss))


    cdef int size = py_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int path_len
    cdef int negative
    cdef int i, j, k
    cdef long result = 0

    # For negative sampling
    cdef np.uint32_t *table = <np.uint32_t *>(np.PyArray_DATA(py_table))
    cdef unsigned long long table_len = len(py_table)
    cdef unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)


    path_len = <int>min(MAX_SENTENCE_LEN, len(py_edge))
    negative = <int>py_negative

    for i in range(path_len):
        word = py_edge[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        next_random = loss_o1_o2(negative, table, table_len, node_embedding, node_embedding,
                                  size, indexes[1], indexes[0], ONEF,  loss, work, next_random)
        next_random = loss_o1_o2(negative, table, table_len, node_embedding, node_embedding,
                                  size, indexes[0], indexes[1], ONEF,  loss, work, next_random)

    return result


def loss_o2(py_node_embedding, py_context_embedding, py_path, py_negative, py_window, py_table,
             py_alpha=1.0, py_size=None, py_work=None, py_loss=None):

    cdef REAL_t *node_embedding = <REAL_t *>(np.PyArray_DATA(py_node_embedding))
    cdef REAL_t *context_embedding = <REAL_t *>(np.PyArray_DATA(py_context_embedding))
    cdef REAL_t *work = <REAL_t *>(np.PyArray_DATA(py_work))
    cdef REAL_t *loss = <REAL_t *>(np.PyArray_DATA(py_loss))


    cdef REAL_t _alpha = py_alpha
    cdef int size = py_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]

    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]

    cdef int path_len
    cdef int negative = py_negative
    cdef int window = py_window
    cdef int i, j, k
    cdef long result = 0

    # For negative sampling
    cdef np.uint32_t *table = <np.uint32_t *>(np.PyArray_DATA(py_table))
    cdef unsigned long long table_len = len(py_table)
    cdef unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)


    path_len = <int>min(MAX_SENTENCE_LEN, len(py_path))

    for i in range(path_len):
        node = py_path[i]
        if node is None:
            codelens[i] = 0
        else:
            indexes[i] = node.index
            reduced_windows[i] = np.random.randint(window)
            codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(path_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > path_len:
                k = path_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                next_random = loss_o1_o2(negative, table, table_len, node_embedding, context_embedding,
                                      size, indexes[i], indexes[j], _alpha, loss, work, next_random)
    return result


def train_o1(py_node_embedding, py_edge, py_lr, py_negative, py_table,
             py_size=None, py_work=None):

    cdef REAL_t *node_embedding = <REAL_t *>(np.PyArray_DATA(py_node_embedding))
    cdef REAL_t *work = <REAL_t *>(np.PyArray_DATA(py_work))

    cdef REAL_t _lr = py_lr

    cdef int size = py_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int path_len
    cdef int negative
    cdef int i, j, k
    cdef long result = 0

    # For negative sampling
    cdef np.uint32_t *table = <np.uint32_t *>(np.PyArray_DATA(py_table))
    cdef unsigned long long table_len = len(py_table)
    cdef unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)


    path_len = <int>min(MAX_SENTENCE_LEN, len(py_edge))
    negative = <int>py_negative

    for i in range(path_len):
        word = py_edge[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        next_random = fast_o1(negative, table, table_len, node_embedding, node_embedding,
                              size, indexes[1], indexes[0], _lr,  work, next_random)
        next_random = fast_o1(negative, table, table_len, node_embedding, node_embedding,
                              size, indexes[0], indexes[1], _lr,  work, next_random)
    return result



def train_o2(py_node_embedding, py_context_embedding, py_path, py_lr, py_negative, py_window, py_table,
             py_alpha=1.0, py_size=None, py_work=None):

    cdef REAL_t *node_embedding = <REAL_t *>(np.PyArray_DATA(py_node_embedding))
    cdef REAL_t *context_embedding = <REAL_t *>(np.PyArray_DATA(py_context_embedding))
    cdef REAL_t *work = <REAL_t *>(np.PyArray_DATA(py_work))

    cdef REAL_t _lr = py_lr
    cdef REAL_t _alpha = py_alpha

    cdef int size = py_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]

    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]

    cdef int path_len
    cdef int negative = py_negative
    cdef int window = py_window
    cdef int i, j, k
    cdef long result = 0

    # For negative sampling
    cdef np.uint32_t *table = <np.uint32_t *>(np.PyArray_DATA(py_table))
    cdef unsigned long long table_len = len(py_table)
    cdef unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)


    path_len = <int>min(MAX_SENTENCE_LEN, len(py_path))

    for i in range(path_len):
        node = py_path[i]
        if node is None:
            codelens[i] = 0
        else:
            indexes[i] = node.index
            reduced_windows[i] = np.random.randint(window)
            # reduced_windows[i] = 0
            codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(path_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > path_len:
                k = path_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                next_random = fast_o2(negative, table, table_len, node_embedding, context_embedding,
                                      size, indexes[i], indexes[j], _lr, _alpha, work, next_random)

    return result

    # for i in range(path_len):
    #     word = py_path[i]
    #     if word is None:
    #         codelens[i] = 0
    #     else:
    #         indexes[i] = word.index
    #         codelens[i] = 1
    #         result += 1
    #
    # # release GIL & train on the sentence
    # with nogil:
    #     for i in range(path_len):
    #         if codelens[i] == 0:
    #             continue
    #         j = i - window
    #         if j < 0:
    #             j = 0
    #         k = i + window + 1
    #         if k > path_len:
    #             k = path_len
    #         for j in range(j, k):
    #             if j == i or codelens[j] == 0:
    #                 continue
    #             else:
    #                 next_random = fast_o2(negative, table, table_len, node_embedding, context_embedding,
    #                                     size, indexes[i], indexes[j], _lr, _alpha,  work, next_random)
    # return result


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    global fast_o1
    global fast_o2
    global fast_community_sdg

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        fast_o1 = fast0_o1
        fast_o2 = fast0_o2
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        fast_o1 = fast1_o1
        fast_o2 = fast1_o2
        return 1  # float

FAST_VERSION = init()  # initialize the module