# encoding=utf8
import os
import numpy as np
import networkx as nx

import scipy as scipy
import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
import scipy.io

from sklearn import preprocessing
import argparse
import time


class RBQR():
    def __init__(self, graph_file, emb_file1, emb_file2, dimension):
        self.graph = graph_file
        self.emb1 = emb_file1
        self.emb2 = emb_file2
        self.dimension = dimension


        data = scipy.io.loadmat(graph_file)
        self.matrix0 = data['network'].astype('float32')
        self.node_number = self.matrix0.shape[0]
        self.randmatrix =  np.random.normal(0, 1./np.sqrt(self.node_number),size=[self.node_number, dimension]).astype(dtype='float32')

        self.node_number = self.matrix0.shape[0]

        print(self.matrix0.shape)

    def get_embedding_rand(self, matrix, dimension, blockSize):
        # Sparse randomized tSVD for fast embedding
        i=0
        matrix_ = matrix
        Q = None
        omg = self.randmatrix
        for j in range(0, 3):
            omg = matrix_ * omg
        omg = np.hsplit(omg, int(dimension/blockSize))
        for i in range(0, int(dimension/blockSize)):
            q,_ = np.linalg.qr(omg[i])
            if i > 0:
                q,_ = np.linalg.qr(q-Q.dot(Q.T.dot(q)))
                b = q.T * matrix_
                Q = np.concatenate((Q, q), 1)
                B = np.concatenate((B, b), 0)
            else:
                Q = q
                B = q.T * matrix_
            i += blockSize
        features_matrix = B/np.linalg.norm(B, axis=0, keepdims=1)
        return features_matrix.T
    # def get_embedding_rand(self, matrix, dimension, blockSize):
    #     # Sparse randomized tSVD for fast embedding
    #     i=0
    #     matrix_ = matrix
    #     Q = None
    #     omg = self.randmatrix
    #     for j in range(0, 3):
    #         omg = matrix_ * omg
    #     omg = np.hsplit(omg, int(dimension/blockSize))
    #     randmatrix = np.hsplit(self.randmatrix, int(dimension / blockSize))
    #     for i in range(0, int(dimension/blockSize)):
    #         if i > 0:
    #             q, _ = np.linalg.qr(omg[i]-Q.dot(B.dot(randmatrix[i])))
    #             q,_ = np.linalg.qr(q-Q.dot(Q.T.dot(q)))
    #             b = q.T * matrix_
    #             Q = np.concatenate((Q, q), 1)
    #             B = np.concatenate((B, b), 0)
    #         else:
    #             q, _ = np.linalg.qr(omg[i])
    #             Q = q
    #             B = q.T * matrix_
    #         i += blockSize
    #     features_matrix = B/np.linalg.norm(B, axis=0, keepdims=1)
    #     return features_matrix.T

    def get_embedding_dense(self, matrix, dimension):
        # get dense embedding via SVD
        t1 = time.time()
        U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = U / np.linalg.norm(U, axis=1, keepdims=1)
        #U = preprocessing.normalize(U, "l2")
        print('densesvd time', time.time() - t1)
        return U

    def pre_factorization(self, tran, mask):
        #Network Embedding as Sparse Matrix Factorization
        t1 = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = scipy.sparse.diags(neg, format="csr")
        neg = mask.dot(neg)
        print("neg", time.time() - t1)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        t1 = time.time()
        # features_matrix = self.rank1_deepwalk_matrix(F,self.dimension)

        features_matrix = self.get_embedding_rand(F, self.dimension, 16)
        print('sparse proximity time', time.time() - t1)
        return features_matrix

    def bksvd(self,A,k,iter=3,bsize=8):
        u = np.zeros(1,A.shape[1])
        l = ones(A.shape[0],1)
        n = A.shape[0]
        K = zeros(A.shape[0],bsize*iter)
        block = np.random.normal(A.shape[1],bsize)
        block,_ = np.linalg.qr(block)
        T = np.zeros(A.shape[1],bsize)

        for i in range(iter):
            T = A*block - l*(u*block)
            block= A.t*T - u.t*(l.t*T)
            block,_ = qr(block)
            K[:, (i - 1) * bsize + 1:i * bsize] = block
        Q,_ = qr(K)

        T = A*Q - l*(u*Q)

        Ut,St,Vt = np.linalg.svd(T)
        S = St[1:k,1:k]
        U = Ut[:,1:k]
        V = Q*Vt[:,1:k]
        return U,S,V
    # def taylor_expansion(self, A, a):
    #     # NE Enhancement via Spectral Propagation
    #     print('taylor Series -----------------')
    #     t1 = time.time()
    #
    #
    #     A = sp.eye(self.node_number,dtype='float32') + A
    #     DA = preprocessing.normalize(A, norm='l1')
    #     L = sp.eye(self.node_number,dtype='float32') - DA
    #
    #     Lx1 = L.dot(a)
    #     Lx1 = 0.5 * L.dot(Lx1) - a
    #
    #     mm = DA.dot(Lx1)
    #     emb = self.get_embedding_dense(mm, self.dimension)
    #     return emb

    def taylor_expansion(self, A, a):
            # NE Enhancement via Spectral Propagation
            print('taylor Series -----------------')
            t1 = time.time()

            A = sp.eye(self.node_number, dtype='float32') + A
            DA = preprocessing.normalize(A, norm='l1')

            Lx1 = DA.dot(a)
            Lx2 = DA.dot(Lx1)
            Lx3 = DA.dot(Lx2)
            mm = Lx3 - 2*Lx2 - Lx1
            emb = self.get_embedding_dense(mm, self.dimension)
            return emb

    # def taylor_expansion(self, A, a):
    #     # NE Enhancement via Spectral Propagation
    #     print('taylor Series -----------------')
    #     t1 = time.time()
    #
    #
    #     A = sp.eye(self.node_number,dtype='float32') + A
    #     DA = preprocessing.normalize(A, norm='l1')
    #     mm = a
    #     Lx1 = DA.dot(a)
    #     mm = Lx1
    #     # Lx1 = DA.dot(Lx1)
    #     # mm+= Lx1
    #     # Lx1 = DA.dot(Lx1)
    #     # mm+= Lx1
    #     emb = self.get_embedding_dense(mm, self.dimension)
    #     return emb




def save_embedding(emb_file, features):
    np.save(emb_file, features, allow_pickle=False)



def parse_args():
    parser = argparse.ArgumentParser(description="Run ProNE.")
    parser.add_argument('-graph', nargs='?', default='data/blogcatalog.mat',
                        help='Graph path')
    parser.add_argument('-emb1', nargs='?', default='emb/blogcatalog.emb',
                        help='Output path of sparse embeddings')
    parser.add_argument('-emb2', nargs='?', default='emb/blogcatalog_spectral.emb',
                        help='Output path of enhanced embeddings')
    parser.add_argument('-dimension', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    return parser.parse_args()


def main():
    args = parse_args()

    t_0 = time.time()
    model = RBQR(args.graph, args.emb1, args.emb2, args.dimension)
    t_1 = time.time()

    features_matrix = model.pre_factorization(model.matrix0, model.matrix0)

    t_2 = time.time()

    embeddings_matrix = model.taylor_expansion(model.matrix0, features_matrix)

    t_3 = time.time()


    print('---', model.node_number)
    print('total time', t_3 - t_0)
    print('sparse NE time', t_2 - t_1)
    print('spectral Pro time', t_3 - t_2)

    save_embedding(args.emb1, features_matrix)
    save_embedding(args.emb2, embeddings_matrix)
    print('save embedding done')


if __name__ == '__main__':
    main()
