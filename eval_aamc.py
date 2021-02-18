import numpy as np
import networkx as nx
import sys
import time
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix
from itertools import chain
import matplotlib
import argparse
import os
import cPickle as pickle
from sklearn import preprocessing

def get_ac(graph, X, y):
    adj = nx.adjacency_matrix(graph)
    P = preprocessing.normalize(adj, norm='l1', axis=1)
    
    XT = X.T
    xsum = X.dot(XT.sum(axis=1))
    xsum[xsum==0]=1
    #X = X/xsum
    xsum = csr_matrix(1.0/xsum)
    X = X.multiply(xsum)
    print(type(X), X.shape)
    
    alpha = 0.2
    beta = 0.35
    t = 5
    
    n = X.shape[0]
    num_cluster = y.max()+1-y.min()
    if(y.min()>0):
        y = y-y.min()
        
    print(n, len(y), num_cluster)
    #print(np.unique(y))
    
    vectors_discrete = csc_matrix((np.ones(len(y)), (np.arange(0, n), y)), shape=(n, num_cluster)).toarray()
    vectors_f = vectors_discrete
    vectors_fs = np.sqrt(vectors_f.sum(axis=0))
    vectors_fs[vectors_fs==0]=1
    vectors_f = vectors_f*1.0/vectors_fs
    q_prime = vectors_f
    h = q_prime
    print(h.shape)
    print(X.shape)
    print(XT.shape)
    print(P.shape)
    for tt in range(t):
        h = (1-alpha)*((1-beta)*P.dot(h)+ (beta)*X.dot(XT.dot(h))) +q_prime
    h = alpha*h
    h = q_prime-h
    
    conductance_cur = 0
    for k in range(num_cluster):
        conductance_cur = conductance_cur + (q_prime[:,k].T).dot(h[:,k])#[0,0]
    
    return conductance_cur/num_cluster

def load_data(args):
    folder = "./data/"
    edge_file = folder+args.data+"/edgelist.txt"
    feature_file = folder+args.data+"/attrs.pkl"

    print("loading from "+feature_file)
    attributes = pickle.load(open(feature_file))
    attributes[attributes>0]=1
    attributes[attributes<0]=0
    attributes=sparse.csr_matrix(attributes)

    n = attributes.shape[0]
    print("loading from "+edge_file)
    graph = nx.read_edgelist(edge_file, create_using=nx.Graph(), nodetype=int)
    for v in range(n):
        graph.add_node(v)

    print("loading from "+args.cfile)
    #lst = []
    #with open(args.cfile, "r") as fin:
    #    for line in fin:
    #        lst.append(int(line.strip()))
            
    print("n=", n)
    lst = []
    with open(args.cfile, "r") as fin:
        lines = fin.readlines()
        print("lines=", len(lines))
        lst = [0]*n#len(lines)
        i=0
        for line in lines:
            l = line.strip().split()
            if len(l)>1:
                v,c=int(l[0]),int(l[1])
                lst[v-1]=c
            else:
                lst[i]=int(l[0])
            i=i+1

    label = np.asarray(lst,dtype=np.int)
    return graph, attributes, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--cfile', type=str, help='cluster file name')
    args = parser.parse_args()
    
    G, attributes, label = load_data(args)

    print(attributes.shape)
    ac = get_ac(G, attributes, label)
    print (ac)

