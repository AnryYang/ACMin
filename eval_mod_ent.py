import numpy as np
import networkx as nx
import sys
import time
from scipy import sparse
from itertools import chain
import matplotlib
import argparse
import os
import cPickle as pickle


def get_cluster(label):
    clusters = dict()
    for i in range(len(label)):
        if label[i] not in clusters:
            clusters[label[i]] = [i]
        else:
            clusters[label[i]].append(i)
    return clusters


def cut_size(G, S, T=None):
    """ modify source code from networkx 2.4
    Returns the size of the cut between two sets of nodes.

    A *cut* is a partition of the nodes of a graph into two sets. The
    *cut size* is the sum of the weights of the edges "between" the two
    sets of nodes.

    Parameters
    ----------
    G : NetworkX graph

    S : sequence
        A sequence of nodes in `G`.

    T : sequence
        A sequence of nodes in `G`. If not specified, this is taken to
        be the set complement of `S`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        Total weight of all edges from nodes in set `S` to nodes in
        set `T` (and, in the case of directed graphs, all edges from
        nodes in `T` to nodes in `S`).

    Examples
    --------
    In the graph with two cliques joined by a single edges, the natural
    bipartition of the graph into two blocks, one for each clique,
    yields a cut of weight one::

        >>> G = nx.barbell_graph(3, 0)
        >>> S = {0, 1, 2}
        >>> T = {3, 4, 5}
        >>> nx.cut_size(G, S, T)
        1

    Each parallel edge in a multigraph is counted when determining the
    cut size::

        >>> G = nx.MultiGraph(['ab', 'ab'])
        >>> S = {'a'}
        >>> T = {'b'}
        >>> nx.cut_size(G, S, T)
        2

    Notes
    -----
    In a multigraph, the cut size is the total weight of edges including
    multiplicity.

    """
    edges = nx.edge_boundary(G, S, T)
    if G.is_directed():
        edges = chain(edges, nx.edge_boundary(G, T, S))
    return sum(1 for u,v in edges)


def modularity(G, labels):
    '''
    Args:
        G (networkx graph): input graph
        labels (ndarray 1*n): label[v] is the cluster label for node v
    Returns:
        modularity (float)
    '''
    clusters = get_cluster(labels)
    mod = 0
    m = G.size()
    for k in clusters:
        fkk = 1.0*G.subgraph(clusters[k]).size()/m # fkk = |Ekk|/|E|
        ak = 0 # ak = \sum_{l=1}^k f_kl = \sum_{l=1}^k |Ekl|/2|E|
        for l in clusters:
            if k == l:
                continue
            ak += cut_size(G=G,S=clusters[k],T=clusters[l]) / m
        mod += (fkk-ak**2)
    return mod

def get_p(R, cluster, attribute):
    '''
    Args:
        R (sparse matrix): sparse node attribute matrix
        cluster (list): a list of node id inside current cluster
        attribute (int): the specified attribute id
    Returns:
        p_{cluster}^{attribute} (float): the fraction of vertices in current
        cluster that take the value 1 in current attribute.
    '''
    #res = 0
    #for each in cluster:
    #    if R[each,attribute]==1:
    #        res+=1
    res = R[cluster,attribute].sum()
    #print(res)
    return 1.0*res/len(cluster)

def entropy(labels, R):
    '''
    Args:
        labels (ndarray 1*n): label[v] is the cluster label for node v
        R (sparse matrix): sparse node attribute matrix
    Returns:
        entropys (dict): key is cluster value is average entropy
    '''
    n,d = R.shape
    entro = dict()
    clusters = get_cluster(labels)
    for each in clusters:
        res = 0
        for i in range(d):
            p = get_p(R,clusters[each],i)
            if p>0 and p<1:
                res -= (p*np.log2(p)+(1-p)*np.log2(1-p))
        entro[each] = len(clusters[each])*res/d/n
    return sum(entro.values())


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

    print("loading from "+args.cfile)
    #lst = []
    #with open(args.cfile, "r") as fin:
    #    for line in fin:
    #        lst.append(int(line.strip()))
            
    lst = []
    with open(args.cfile, "r") as fin:
        lines = fin.readlines()
        lst = [0]*len(lines)
        i=0
        for line in lines:
            l = line.strip().split()
            if len(l)>1:
                v,c=int(l[0]),int(l[1])
                lst[v-1]=c
            else:
                lst[i]=int(l[0])
            i=i+1

    label = np.asarray(lst)
    return graph, attributes, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--cfile', type=str, help='cluster file name')
    args = parser.parse_args()
    
    G, attributes, label = load_data(args)

    mod = modularity(G=G,labels=label)
    print (mod)

    #print(attributes.shape)
    #entro = entropy(labels=label, R=attributes)
    #print (entro)

