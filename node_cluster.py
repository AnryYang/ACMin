#########################################################################
# File Name: node_cluster.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Fri 05 Apr 2019 04:33:10 PM
#########################################################################
#!/usr/bin/env/ python

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import metrics
import numpy as np
import argparse
import os
import cPickle as pickle
import networkx as nx
from scipy.sparse.linalg import svds
import scipy.sparse as sp
from scipy.sparse import identity
from scipy import linalg
from scipy import sparse
from munkres import Munkres
from sklearn import preprocessing
from sklearn.decomposition import NMF
import heapq
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from spectral import discretize
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from scipy.sparse.linalg.eigen.arpack import eigs as largest_eigs
from scipy.linalg import qr
from scipy.linalg import orth
from scipy.sparse.csgraph import laplacian
import time
import sklearn
from sklearn.linear_model import SGDRegressor
from scipy.sparse import csc_matrix, csr_matrix
from numpy import linalg as LA
import operator
import random

print(sklearn.__version__)

def read_cluster(N,file_name):
    if not file_name or not os.path.exists(file_name):
        raise Exception("label file not exist!")
    f = open(file_name, "r")
    lines = f.readlines()
    f.close()
    #N = len(lines)
    y = np.zeros(N, dtype=int)
    for line in lines:
        i, l = line.strip("\n\r").split()
        i, l = int(i), int(l)
        y[i] = l
    
    return y

class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        print(len(self.true_label), len(self.pred_label))
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal!!!!')
            c1_clusters = {c: set() for c in set(l1)}
            c2_clusters = {c: set() for c in set(l2)}
            
            for i in range(len(self.true_label)):
                c1 = self.true_label[i]
                c2 = self.pred_label[i]
                c1_clusters[c1].add(i)
                c2_clusters[c2].add(i)

            c2_c1 = {}
            for c2 in set(l2):
                for c1 in set(l1):
                    c2_c1[str(c2)+","+str(c1)]=0


            for (c1, s1) in c1_clusters.items():
                for (c2, s2) in c2_clusters.items():
                    num_com_s1s2 = len(s1.intersection(s2))
                    c2_c1[str(c2)+","+str(c1)]=num_com_s1s2

            sorted_x = sorted(c2_c1.items(), key=operator.itemgetter(1), reverse=True)
            
            c2_c1_map = {}
            c1_flag = {c: True for c in set(l1)}
            c2_flag = {c: True for c in set(l2)}
            for (k, v) in sorted_x:
                if len(c2_c1_map.keys())==numclass1:
                    break
                c2, c1 = k.split(',')
                c2, c1 = int(c2), int(c1)
                #print(c2, c1, v)
                if c1_flag[c1] and c2_flag[c2]:
                    c2_c1_map[c2]=c1

                c1_flag[c1] = False
                c2_flag[c2] = False
            
            new_predict = np.zeros(len(self.pred_label))
            for i in range(len(l2)):
                new_predict[i] = c2_c1_map[self.pred_label[i]]
                
        else:
            cost = np.zeros((numclass1, numclass2), dtype=int)
            for i, c1 in enumerate(l1):
                mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
                for j, c2 in enumerate(l2):
                    mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                    cost[i][j] = len(mps_d)

            # match two clustering results by Munkres algorithm
            m = Munkres()
            cost = cost.__neg__().tolist()

            indexes = m.compute(cost)

            # get the match results
            new_predict = np.zeros(len(self.pred_label))
            for i, c in enumerate(l1):
                # correponding label in l2:
                c2 = l2[indexes[i][1]]

                # ai is the index with label==c2 in the pred_label list
                ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
                new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc = self.clusteringAcc()

        return acc, nmi, adjscore

def load_data(args):
    folder = "./data/"
    edge_file = folder+args.data+"/edgelist.txt"
    feature_file = folder+args.data+"/attrs.pkl"
    label_file = folder+args.data + '/labels.txt'

    print("loading from "+feature_file)
    features = pickle.load(open(feature_file))
    
    print("nnz:", features.getnnz())
    print(features.shape)
    n = features.shape[0]
    print("loading from "+edge_file)
    graph = nx.read_edgelist(edge_file, create_using=nx.Graph(), nodetype=int)
    for v in range(n):
        graph.add_node(v)

    print("loading from "+label_file)
    true_clusters = read_cluster(n,label_file)
    return graph, features, true_clusters
    

def si_eig(P, X, alpha, beta, k, a):
    t = 500
    q, _ = qr(a, mode='economic')
    XT = X.T
    xsum = X.dot(XT.sum(axis=1))
    xsum[xsum==0]=1
    X = X/xsum
    for i in range(t):
        z = (1-alpha-beta)*P.dot(q)+ (beta)*X.dot(XT.dot(q))
        p = q
        q, _ = qr(z, mode='economic')
        if np.linalg.norm(p-q, ord=1)<0.01:
            print("converged")
            break
    
    return q


def base_cluster(graph, X, num_cluster,true_clusters):
    print("attributed transition matrix constrcution...")
    adj = nx.adjacency_matrix(graph)
    P = preprocessing.normalize(adj, norm='l1', axis=1)
    n = P.shape[0]
    print(P.shape)

    start_time = time.time()
    alpha=0.2
    beta=0.35
    XX = X.dot(X.T)
    XX = preprocessing.normalize(XX, norm='l1', axis=1)
    PP = (1-beta)*P + beta*XX
    I = identity(n) 
    S = I
    t = 5 #int(1.0/alpha)
    for i in range(t):
        S = (1-alpha)*PP.dot(S)+I

    S = alpha*S
    q = np.zeros(shape=(n,num_cluster))

    predict_clusters = n*[1]
    lls = [i for i in range(num_cluster)]
    for i in range(n):
        ll = random.choice(lls)
        predict_clusters[i] = ll

    M = csc_matrix((np.ones(len(predict_clusters)), (np.arange(0, n), predict_clusters)),shape=(n,num_cluster+1))
    M = M.todense()

    Mss = np.sqrt(M.sum(axis=0))
    Mss[Mss==0]=1
    q = M*1.0/Mss

    largest_evc = np.ones(shape = (n,1))*(1.0/np.sqrt(n*1.0))
    q = np.hstack([largest_evc,q])

    XT = X.T
    xsum = X.dot(XT.sum(axis=1))
    xsum[xsum==0]=1
    xsum = csr_matrix(1.0/xsum)
    X = X.multiply(xsum)
    print(type(X), X.shape)

    predict_clusters = np.asarray(predict_clusters,dtype=np.int)
    print(q.shape)

    epsilon_f = 0.005
    tmax = 200
    err = 1
    for i in range(tmax):
        z = S.dot(q)
        q_prev = q
        q, _ = qr(z, mode='economic')

        err = LA.norm(q-q_prev)/LA.norm(q)
        if err <= epsilon_f:
            break
        
        if i==tmax-1:
            evecs_large_sparse = q
            evecs_large_sparse = evecs_large_sparse[:,1:num_cluster+1]

            kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_jobs=-1, algorithm='full', init='random', n_init=1, max_iter=50).fit(evecs_large_sparse)
            predict_clusters = kmeans.predict(evecs_large_sparse)


    time_elapsed = time.time() - start_time
    print("%f seconds are taken to train"%time_elapsed)

    return predict_clusters 
    
def get_ac(P, X, XT, y, alpha, beta, t):
    n = X.shape[0]
    num_cluster = y.max()+1-y.min()
    if(y.min()>0):
        y = y-y.min()
        
    print(n, len(y), num_cluster)
    
    vectors_discrete = csc_matrix((np.ones(len(y)), (np.arange(0, n), y)), shape=(n, num_cluster)).toarray()
    vectors_f = vectors_discrete
    vectors_fs = np.sqrt(vectors_f.sum(axis=0))
    vectors_fs[vectors_fs==0]=1
    vectors_f = vectors_f*1.0/vectors_fs
    q_prime = vectors_f
    h = q_prime
    for tt in range(t):
        h = (1-alpha)*((1-beta)*P.dot(h)+ (beta)*X.dot(XT.dot(h))) +q_prime
    h = alpha*h
    h = q_prime-h
    
    conductance_cur = 0
    for k in range(num_cluster):
        conductance_cur = conductance_cur + (q_prime[:,k].T).dot(h[:,k])#[0,0]
    
    return conductance_cur/num_cluster


def cluster(graph, X, num_cluster,true_clusters, alpha=0.2, beta = 0.35, t=5, tmax=200, ri=False):
    print("attributed transition matrix constrcution...")
    adj = nx.adjacency_matrix(graph)
    P = preprocessing.normalize(adj, norm='l1', axis=1)
    n = P.shape[0]
    print(P.shape)

    epsilon_r = 6*n*np.log(n*1.0)/X.getnnz()
    print("epsilon_r threshold:", epsilon_r)

    degrees = dict(graph.degree())
    topk_deg_nodes = heapq.nlargest(5*t*num_cluster, degrees, key=degrees.get)
    PC = P[:,topk_deg_nodes]
    M = PC
    for i in range(t-1):
        M = (1-alpha)*P.dot(M)+PC
    
    class_evdsum = M.sum(axis=0).flatten().tolist()[0]
    newcandidates = np.argpartition(class_evdsum, -num_cluster)[-num_cluster:]
    M = M[:,newcandidates]    
    labels = np.argmax(M, axis=1).flatten().tolist()[0]
    labels = np.asarray(labels,dtype=np.int)
    
    # random initialization
    if ri is True:
        lls = np.unique(labels)
        for i in range(n):
            ll = random.choice(lls)
            labels[i] = ll
        
    M = csc_matrix((np.ones(len(labels)), (np.arange(0, M.shape[0]), labels)),shape=(M.shape))
    M = M.todense()

    start_time = time.time()
    
    print("eigen decomposition...")

    Mss = np.sqrt(M.sum(axis=0))
    Mss[Mss==0]=1
    q = M*1.0/Mss

    largest_evc = np.ones(shape = (n,1))*(1.0/np.sqrt(n*1.0))
    q = np.hstack([largest_evc,q])

    XT = X.T
    xsum = X.dot(XT.sum(axis=1))
    xsum[xsum==0]=1
    xsum = csr_matrix(1.0/xsum)
    X = X.multiply(xsum)
    print(type(X), X.shape)
    predict_clusters_best=labels
    iter_best = 0
    conductance_best=100
    conductance_best_acc = [0]*3
    acc_best = [0]*3
    acc_best_iter = 0
    acc_best_conductance = 0
    epsilon_f = 0.005
    err = 1
    
    for i in range(tmax):
        z = (1-beta)*P.dot(q)+ (beta)*X.dot(XT.dot(q))
        q_prev = q
        q, _ = qr(z, mode='economic')
        
        err = LA.norm(q-q_prev)/LA.norm(q)

        if (i+1)%20==0:
            evecs_large_sparse = q
            evecs_large_sparse = evecs_large_sparse[:,1:num_cluster+1]
            predict_clusters, q_prime = discretize(evecs_large_sparse)
            
            conductance_cur = 0
            h = q_prime
            for tt in range(1):
                h = (1-alpha)*((1-beta)*P.dot(h)+ (beta)*X.dot(XT.dot(h))) +q_prime
            h = alpha*h
            h = q_prime-h
            
            for k in range(num_cluster):
                conductance_cur = conductance_cur + (q_prime[:,k].T).dot(h[:,k])#[0,0]
            conductance_cur=conductance_cur/num_cluster
                
            if conductance_cur<conductance_best:
                conductance_best = conductance_cur
                predict_clusters_best = predict_clusters
                iter_best = i
        
            print(i, err, conductance_cur)

        if err <= epsilon_f:
            break
    
    if tmax==0:
        evecs_large_sparse = q
        evecs_large_sparse = evecs_large_sparse[:,1:num_cluster+1]
        predict_clusters, q_prime = discretize(evecs_large_sparse)
        predict_clusters_best = predict_clusters
    
    time_elapsed = time.time() - start_time
    print("%f seconds are taken to train"%time_elapsed)
    print(np.unique(predict_clusters_best))
    print("best iter: %d, best condutance: %f, acc: %f, %f, %f"%(iter_best, conductance_best, conductance_best_acc[0], conductance_best_acc[1], conductance_best_acc[2]))
    return predict_clusters_best
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--k', type=int, default=0, help='the number of clusters')
    args = parser.parse_args() 
    
    print("loading data ", args.data)
    graph, feats, true_clusters = load_data(args)
    
    n = feats.shape[0]
    if args.k>0:
        num_cluster = args.k
    else:
        num_cluster = len(np.unique(true_clusters))

    print("k=", num_cluster)
    
    alpha = 0.2
    beta = 0.35
    t = 5
    tmax = 200

    predict_clusters = cluster(graph, feats, num_cluster, true_clusters, alpha, beta, t, tmax, False)

    if args.k<=0:
        cm = clustering_metrics(true_clusters, predict_clusters)
        print("%f\t%f\t%f"%cm.evaluationClusterModelFromLabel())

    print("-------------------------------")
    K = len(set(predict_clusters))
    with open("sc."+args.data+"."+str(K)+".cluster.txt", "w") as fout:
        for i in range(len(predict_clusters)):
            fout.write(str(predict_clusters[i])+"\n")
