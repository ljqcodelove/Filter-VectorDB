import pdb
import pickle
import numpy as np
import os
import psutil
from memory_profiler import profile
import hnswlib
from scipy.sparse import csr_matrix
import gc

from multiprocessing.pool import ThreadPool

import faiss

from neurips23.filter.base import BaseFilterANN
from benchmark.datasets import DATASETS
from benchmark.dataset_io import download_accelerated

import bow_id_selector

# import uint8_knn

import time

def csr_get_row_indices(m, i):
    """ get the non-0 column indices for row i in matrix m """
    return m.indices[m.indptr[i]: m.indptr[i + 1]]


def make_bow_id_selector(mat, id_mask=0):
    sp = faiss.swig_ptr
    if id_mask == 0:
        return bow_id_selector.IDSelectorBOW(mat.shape[0], sp(mat.indptr), sp(mat.indices))
    else:
        return bow_id_selector.IDSelectorBOWBin(
            mat.shape[0], sp(mat.indptr), sp(mat.indices), id_mask
        )


def set_invlist_ids(invlists, l, ids):
    n, = ids.shape
    ids = np.ascontiguousarray(ids, dtype='int64')
    assert invlists.list_size(l) == n
    faiss.memcpy(
        invlists.get_ids(l),
        faiss.swig_ptr(ids), n * 8
    )


def csr_to_bitcodes(matrix, bitsig):
    """ Compute binary codes for the rows of the matrix: each binary code is
    the OR of bitsig for non-0 entries of the row.
    """
    indptr = matrix.indptr
    indices = matrix.indices
    n = matrix.shape[0]
    bit_codes = np.zeros(n, dtype='int64')
    for i in range(n):
        # print(bitsig[indices[indptr[i]:indptr[i + 1]]])
        bit_codes[i] = np.bitwise_or.reduce(bitsig[indices[indptr[i]:indptr[i + 1]]])
    return bit_codes

def read_sparse_matrix_fields(fname):
    """ read the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype='float32', count=nnz)
        return data, indices, indptr, ncol

def mmap_sparse_matrix_fields(fname):
    """ mmap the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
    ofs = sizes.nbytes
    indptr = np.memmap(fname, dtype='int64', mode='r', offset=ofs, shape=nrow + 1)
    ofs += indptr.nbytes
    indices = np.memmap(fname, dtype='int32', mode='r', offset=ofs, shape=nnz)
    ofs += indices.nbytes
    data = np.memmap(fname, dtype='float32', mode='r', offset=ofs, shape=nnz)
    return data, indices, indptr, ncol

def read_sparse_matrix(fname, do_mmap=False):
    """ read a CSR matrix in spmat format, optionally mmapping it instead """
    if not do_mmap:
        data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    else:
        data, indices, indptr, ncol = mmap_sparse_matrix_fields(fname)

    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))

class BinarySignatures:
    """ binary signatures that encode vectors """

    def __init__(self, meta_b, proba_1):
        nvec, nword = meta_b.shape
        # number of bits reserved for the vector ids
        self.id_bits = int(np.ceil(np.log2(nvec)))
        # number of bits for the binary signature
        self.sig_bits = nbits = 63 - self.id_bits

        # select binary signatures for the vocabulary
        rs = np.random.RandomState(123)  # we rely on this to be reproducible!
        bitsig = np.packbits(rs.rand(nword, nbits) < proba_1, axis=1)
        bitsig = np.pad(bitsig, ((0, 0), (0, 8 - bitsig.shape[1]))).view("int64").ravel()
        self.bitsig = bitsig

        # signatures for all the metadata matrix
        self.db_sig = csr_to_bitcodes(meta_b, bitsig) << self.id_bits

        # mask to keep only the ids
        self.id_mask = (1 << self.id_bits) - 1

    def query_signature(self, w1, w2):
        """ compute the query signature for 1 or 2 words """
        sig = self.bitsig[w1]
        if w2 != -1:
            sig |= self.bitsig[w2]
        return int(sig << self.id_bits)


class FAISS(BaseFilterANN):

    def __init__(self, metric, index_params):
        self._index_params = index_params
        self._metric = metric
        print(index_params)
        self.indexkey = index_params.get("indexkey", "IVF32768,SQ8")
        self.binarysig = index_params.get("binarysig", True)
        self.binarysig_proba1 = index_params.get("binarysig_proba1", 0.1)
        self.metadata_threshold = 1e-3
        self.nt = index_params.get("threads", 1)
        self.N = 10000000
        self.initP = None


    @profile
    def fit(self, dataset):

        memoryNow = psutil.Process().memory_info().rss / 1024
        ds = DATASETS[dataset]()

        self.meta_b = ds.get_dataset_metadata()
        self.meta_b.sort_indices()

        xb = ds.get_dataset()
        self.nb = ds.nb
        self.xb = xb

        self.initP = hnswlib.Index(space='l2', dim=192)
        num_elements = len(self.xb)
        self.N = num_elements

        self.initP.init_index(max_elements=num_elements, ef_construction=50, M=10)
        self.initP.set_ef(20)
        self.initP.set_num_threads(8)

        meta_b = self.meta_b

        def set_initial_label(q):
            self.initP.set_filter_labels(csr_get_row_indices(meta_b, q), q, len(csr_get_row_indices(meta_b, q)))

        # for i in range(num_elements):
        #     if (i % 50000 == 0):
        #         print("check log: ", i/50000)
        #     self.initP.set_filter_labels(csr_get_row_indices(meta_b, i), i, len(csr_get_row_indices(meta_b, i)))

        # pool = ThreadPool(self.nt)
        # list(pool.map(set_initial_label, range(num_elements)))

        # print(csr_get_row_indices(meta_b, 0))
        # print(len(csr_get_row_indices(meta_b, 0)))
        self.initP.add_items(self.xb, np.arange(num_elements))
        print("build end")


    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.faissindex"

    def binarysig_name(self, name):
        return f"data/{name}.{self.indexkey}.binarysig"

    def load_index(self, dataset):
        """
        Load the index for dataset. Returns False if index
        is not available, True otherwise.

        Checking the index usually involves the dataset name
        and the index build paramters passed during construction.
        """
        if not os.path.exists(self.index_name(dataset)):
            if 'url' not in self._index_params:
                return False

            print('Downloading index in background. This can take a while.')
            download_accelerated(self._index_params['url'], self.index_name(dataset), quiet=True)

        print("Loading index")

        self.index = faiss.read_index(self.index_name(dataset))

        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)

        ds = DATASETS[dataset]()

        if ds.search_type() == "knn_filtered" and self.binarysig:
            if not os.path.exists(self.binarysig_name(dataset)):
                print("preparing binary signatures")
                meta_b = ds.get_dataset_metadata()
                # self.binsig = BinarySignatures(meta_b, self.binarysig_proba1)
            else:
                print("loading binary signatures")
                # self.binsig = pickle.load(open(self.binarysig_name(dataset), "rb"))
        else:
            self.binsig = None

        if ds.search_type() == "knn_filtered":
            self.meta_b = ds.get_dataset_metadata()
            self.meta_b.sort_indices()

        self.nb = ds.nb
        self.xb = ds.get_dataset()

        X = ds.get_queries()
        metadata = ds.get_queries_metadata()
        nq = X.shape[0]
        meta_q = metadata


        meta_b = self.meta_b
        docs_per_word = meta_b.T.tocsr()
        ndoc_per_word = docs_per_word.indptr[1:] - docs_per_word.indptr[:-1]
        freq_per_word = ndoc_per_word / ds.nb

        cnt = 0

        self.initP = hnswlib.Index(space='l2', dim=192)
        num_elements = len(self.xb)
        self.initP.init_index(max_elements=num_elements, ef_construction=1, M=0)
        self.initP.set_ef(1)
        self.initP.set_num_threads(8)
        self.initP.add_only_data_items(self.xb, np.arange(num_elements))

        self.index_set.append(self.initP)
        cnt += 1

        print("nq is ", nq)
        for q in range(nq):

            qwords = csr_get_row_indices(meta_q, q)
            assert qwords.size in (1, 2)
            w1 = qwords[0]
            freq = freq_per_word[w1]

            if qwords.size == 2:
                w2 = qwords[1]
                if freq_per_word[w2] < freq:
                    freq = freq_per_word[w2]
            else:
                w2 = -1

            if w1 not in self.st and w2 == -1 and freq >= 0.003:
                self.st.add(w1)
                docs = csr_get_row_indices(docs_per_word, w1)
                dim = 192
                num_elements = len(docs)

                data = ds.get_dataset()[docs]

                print("build i-th index and length is ", cnt, len(docs))

                p = hnswlib.Index(space='l2', dim=dim)

                p.init_index(max_elements=num_elements, ef_construction=50, M=5)

                p.set_ef(20)
                p.set_num_threads(8)
                p.set_only_data_level0_memory(index=self.index_set[0])

                p.add_items_new(data, docs)
                p.clear_lock()

                self.mp[int(w1)] = cnt
                cnt += 1


                self.index_set.append(p)
            elif w2 != -1:
                if w1 > w2:
                    min_w = w2
                    max_w = w1
                else:
                    min_w = w1
                    max_w = w2

                if (int(min_w), int(max_w)) not in self.two_tags_st:
                    docs = csr_get_row_indices(docs_per_word, w1)

                    docs = bow_id_selector.intersect_sorted(
                        docs, csr_get_row_indices(docs_per_word, w2))

                    if len(docs) < self.nb * 0.003:
                        continue

                    self.two_tags_st.add((int(min_w), int(max_w)))

                    print("build i-th two-tags index and length is ", cnt, len(docs))

                    dim = 192
                    num_elements = len(docs)

                    data = ds.get_dataset()[docs]

                    p = hnswlib.Index(space='l2', dim=dim)

                    p.init_index(max_elements=num_elements, ef_construction=50, M=5)

                    p.set_ef(20)
                    p.set_num_threads(8)

                    p.set_only_data_level0_memory(index=self.index_set[0])
                    p.add_items_new(data, docs)
                    p.clear_lock()

                    # p.add_items(data, docs)

                    self.two_tags_mp[(int(min_w), int(max_w))] = cnt
                    cnt += 1


                    self.index_set.append(p)

        return True

    def index_files_to_store(self, dataset):
        """
        Specify a triplet with the local directory path of index files,
        the common prefix name of index component(s) and a list of
        index components that need to be uploaded to (after build)
        or downloaded from (for search) cloud storage.

        For local directory path under docker environment, please use
        a directory under
        data/indices/track(T1 or T2)/algo.__str__()/DATASETS[dataset]().short_name()
        """
        raise NotImplementedError()

    def query(self, X, k):
        nq = X.shape[0]
        self.I = -np.ones((nq, k), dtype='int32')
        bs = 1024
        for i0 in range(0, nq, bs):
            _, self.I[i0:i0 + bs] = self.index.search(X[i0:i0 + bs], k)

    def filtered_query(self, X, filter, k):
        print('running filtered query')
        print('metadata_threshold is ', self.metadata_threshold)
        print('self.N is ', self.N)
        nq = X.shape[0]
        self.I = -np.ones((nq, k), dtype='int32')
        meta_b = self.meta_b
        meta_q = filter
        docs_per_word = meta_b.T.tocsr()
        ndoc_per_word = docs_per_word.indptr[1:] - docs_per_word.indptr[:-1]
        freq_per_word = ndoc_per_word / self.nb

        def process_one_row(q):
            faiss.omp_set_num_threads(1)
            qwords = csr_get_row_indices(meta_q, q)
            assert qwords.size in (1, 2)
            w1 = qwords[0]
            freq = freq_per_word[w1]
            docs = csr_get_row_indices(docs_per_word, w1)

            if qwords.size == 2:
                w2 = qwords[1]
                if freq > freq_per_word[w2]:
                    freq = freq_per_word[w2]
                # freq *= freq_per_word[w2]

                docs = bow_id_selector.intersect_sorted(
                    docs, csr_get_row_indices(docs_per_word, w2))
            else:
                w2 = -1

            if w2 == -1 and len(docs) < self.N * self.metadata_threshold:
                # metadata first

                assert len(docs) >= k, pdb.set_trace()
                xb_subset = self.xb[docs]
                _, Ii = faiss.knn(X[q: q + 1], xb_subset, k=k)

                self.I[q, :] = docs[Ii.ravel()]
            elif w2 == -1 and len(docs) >= self.N * self.metadata_threshold:
                labels, distances = self.initP.knn_query_one_stage(X[q: q + 1], docs, len(docs), k=k, w1=w1, w2=w2)
                labels = labels.ravel()
                self.I[q, :] = labels


            elif w2 != -1:
                if len(docs) >= self.N * self.metadata_threshold:
                    labels, distances = self.initP.knn_query_one_stage(X[q: q + 1], docs, len(docs), k=k, w1=w1, w2=w2)
                    labels = labels.ravel()
                    self.I[q, :] = labels
                else:


                    assert len(docs) >= k, pdb.set_trace()
                    xb_subset = self.xb[docs]
                    _, Ii = faiss.knn(X[q: q + 1], xb_subset, k=k)

                    self.I[q, :] = docs[Ii.ravel()]

        if self.nt <= 1:
            for q in range(nq):
                process_one_row(q)
        else:
            faiss.omp_set_num_threads(self.nt)
            pool = ThreadPool(self.nt)
            list(pool.map(process_one_row, range(nq)))

    # def filtered_query(self, X, filter, k):
    #     print('running filtered query')
    #     nq = X.shape[0]
    #     self.I = -np.ones((nq, k), dtype='int32')
    #     meta_b = self.meta_b
    #     meta_q = filter
    #     docs_per_word = meta_b.T.tocsr()
    #     ndoc_per_word = docs_per_word.indptr[1:] - docs_per_word.indptr[:-1]
    #     freq_per_word = ndoc_per_word / self.nb
    #     query_dict = {}
    #
    #     def block(q):
    #         # 按照关键字进行分块，将同类关键字的query放在一起
    #         # 1. 按照关键字进行分块,使用字典存储
    #         # 创建一个字典，qwords为关键字，value为q
    #         qwords = csr_get_row_indices(meta_q, q)
    #         assert qwords.size in (1, 2)
    #         # 求关键字的频率
    #         w1 = qwords[0]
    #         freq = freq_per_word[w1]
    #         if qwords.size == 2:
    #             w2 = qwords[1]
    #             if freq > freq_per_word[w2]:
    #                 freq = freq_per_word[w2]
    #             # freq *= freq_per_word[w2]
    #         else:
    #             w2 = -1
    #         if freq < self.metadata_threshold:
    #             # 追加进字典，qwords为关键字，value为q
    #             if qwords.size == 2:
    #                 key = (qwords[1], qwords[0])
    #             else:
    #                 key = (qwords[0], -1)
    #             if key in query_dict:
    #                 query_dict[key].append(q)
    #             else:
    #                 query_dict[key] = [q]
    #         elif w2 == -1:
    #             idNow = self.mp[int(w1)]
    #             p = self.index_set[idNow]
    #             # labels, distances = p.knn_query(X[q: q + 1], k=k)
    #             labels, distances = p.knn_query_new(X[q: q + 1], k=k)
    #             labels = labels.ravel()
    #             self.I[q] = labels
    #
    #         else:
    #             if w1 > w2:
    #                 min_w = w2
    #                 max_w = w1
    #             else:
    #                 min_w = w1
    #                 max_w = w2
    #
    #             if (int(min_w), int(max_w)) in self.two_tags_st:
    #                 idNow = self.two_tags_mp[(int(min_w), int(max_w))]
    #                 p = self.index_set[idNow]
    #                 # labels, distances = p.knn_query(X[q: q + 1], k=k)
    #                 labels, distances = p.knn_query_new(X[q: q + 1], k=k)
    #                 # labels, distances = p.knn_query(np.array(X[q: q + 1]), k=k)
    #                 labels = labels.ravel()
    #                 self.I[q, :] = labels
    #
    #     def process(qwords):
    #         w1 = qwords[0]
    #         if qwords[1] != -1:
    #             w2 = qwords[1]
    #         else:
    #             w2 = -1
    #         # metadata first
    #         docs = csr_get_row_indices(docs_per_word, w1)
    #         if w2 != -1:
    #             docs = bow_id_selector.intersect_sorted(
    #                 docs, csr_get_row_indices(docs_per_word, w2))
    #
    #         assert len(docs) >= k, pdb.set_trace()
    #         xb_subset = self.xb[docs]
    #         # 构造query二维矩阵，其是一个二维的numpy数组
    #         # query = []
    #         # for i in query_dict[qwords]:
    #         #     query.append(X[i : i + 1])
    #         # query = np.concatenate(query, axis=0)
    #         # query = np.array([X[i : i + 1] for i in query_dict[qwords]])
    #         query = np.vstack([X[i: i + 1] for i in query_dict[qwords]])
    #         # query = np.concatenate([X[i : i + 1] for i in query_dict[qwords]], axis=0)
    #         # print("query", query.shape, xb_subset.shape)
    #         # _, Ii = faiss.knn(query, xb_subset, k=k)
    #
    #         # 判断query的大小，如果小于32，则使用fbknn，否则使用faiss.knn
    #         # if query.shape[0] < 16:
    #         # _, Ii = faiss.knn(query, xb_subset, k=k)
    #         Ii = uint8_knn.knn(query, xb_subset, k).reshape(query.shape[0], k)
    #         # else:
    #         #     # print("fbknn")
    #         # # print ("query", query.shape, xb_subset.shape)
    #         #     Ii = fbknn.knn(query, xb_subset, k=k)
    #         # self.I[query_dict[key], :] = docs[Ii.ravel()]
    #         for i, q in enumerate(query_dict[qwords]):
    #             self.I[q] = docs[Ii[i]]
    #
    #     faiss.omp_set_num_threads(self.nt)
    #     pool = ThreadPool(self.nt)
    #     # list(pool.map(block, range(nq)))
    #     pool.map(block, range(nq))
    #     # print("query_dict", query_dict.keys())
    #     # 对query_dict进行排序处理，保证两个关键字（a,b)跟在a在一起
    #     # query_dict = dict(sorted(query_dict.items(), key=lambda x: x[0][0]))
    #     # print("query_dict", query_dict.keys())
    #     pool.map(process, query_dict.keys())

    # def filtered_query(self, X, filter, k):
    #     print('running filtered query')
    #     nq = X.shape[0]
    #     self.I = -np.ones((nq, k), dtype='int32')
    #     meta_b = self.meta_b
    #     meta_q = filter
    #     docs_per_word = meta_b.T.tocsr()
    #     ndoc_per_word = docs_per_word.indptr[1:] - docs_per_word.indptr[:-1]
    #     freq_per_word = ndoc_per_word / self.nb
    #
    #     def process_one_row(q):
    #         faiss.omp_set_num_threads(1)
    #         qwords = csr_get_row_indices(meta_q, q)
    #         assert qwords.size in (1, 2)
    #         w1 = qwords[0]
    #         freq = freq_per_word[w1]
    #         if qwords.size == 2:
    #             w2 = qwords[1]
    #             if freq > freq_per_word[w2]:
    #                 freq = freq_per_word[w2]
    #             # freq *= freq_per_word[w2]
    #         else:
    #             w2 = -1
    #
    #         if w2 == -1 and freq < self.metadata_threshold:
    #             # metadata first
    #             docs = csr_get_row_indices(docs_per_word, w1)
    #
    #             assert len(docs) >= k, pdb.set_trace()
    #             xb_subset = self.xb[docs]
    #             _, Ii = faiss.knn(X[q: q + 1], xb_subset, k=k)
    #
    #             self.I[q, :] = docs[Ii.ravel()]
    #         elif w2 == -1 and freq >= self.metadata_threshold:
    #             idNow = self.mp[int(w1)]
    #             p = self.index_set[idNow]
    #             # labels, distances = p.knn_query(X[q: q + 1], k=k)
    #             labels, distances = p.knn_query_new(X[q: q + 1], k=k)
    #             labels = labels.ravel()
    #             self.I[q] = labels
    #
    #         elif w2 != -1:
    #             if w1 > w2:
    #                 min_w = w2
    #                 max_w = w1
    #             else:
    #                 min_w = w1
    #                 max_w = w2
    #
    #             if (int(min_w), int(max_w)) in self.two_tags_st:
    #                 idNow = self.two_tags_mp[(int(min_w), int(max_w))]
    #                 p = self.index_set[idNow]
    #                 # labels, distances = p.knn_query(X[q: q + 1], k=k)
    #                 labels, distances = p.knn_query_new(X[q: q + 1], k=k)
    #                 # labels, distances = p.knn_query(np.array(X[q: q + 1]), k=k)
    #                 labels = labels.ravel()
    #                 self.I[q, :] = labels
    #             else:
    #                 docs = csr_get_row_indices(docs_per_word, w1)
    #
    #                 docs = bow_id_selector.intersect_sorted(
    #                     docs, csr_get_row_indices(docs_per_word, w2))
    #
    #                 assert len(docs) >= k, pdb.set_trace()
    #                 xb_subset = self.xb[docs]
    #                 _, Ii = faiss.knn(X[q: q + 1], xb_subset, k=k)
    #
    #                 self.I[q, :] = docs[Ii.ravel()]
    #
    #     if self.nt <= 1:
    #         for q in range(nq):
    #             process_one_row(q)
    #     else:
    #         faiss.omp_set_num_threads(self.nt)
    #         pool = ThreadPool(self.nt)
    #         list(pool.map(process_one_row, range(nq)))

    def get_results(self):
        return self.I

    def set_query_arguments(self, query_args):
        faiss.cvar.indexIVF_stats.reset()
        self.qas = query_args
        if "ef_construction" in query_args:
            self.ef_construction = query_args['ef_construction']
        else:
            self.ef_construction = 100
        if "ef_search" in query_args:
            self.ef_search = query_args['ef_search']
        else:
            self.ef_search = 100
        if "build_threshold" in query_args:
            self.build_threshold = query_args['build_threshold']
        else:
            self.build_threshold = 0.001
        if "M" in query_args:
            self.M = query_args['M']
        else:
            self.M = 25
        # if "nprobe" in query_args:
        #     self.nprobe = query_args['nprobe']
        #     self.ps.set_index_parameters(self.index, f"nprobe={query_args['nprobe']}")
        #     self.qas = query_args
        # else:
        #     self.nprobe = 1
        if "mt_threshold" in query_args:
            self.metadata_threshold = query_args['mt_threshold']
        else:
            self.metadata_threshold = 1e-3

    def __str__(self):
        return f'Faiss({self.indexkey, self.qas})'

