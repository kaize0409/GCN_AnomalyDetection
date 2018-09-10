import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io
import inspect
import tensorflow as tf
from preprocessing import preprocess_graph, sparse_to_tuple

flags = tf.app.flags
FLAGS = flags.FLAGS



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
# def load_data(dataset):
#     # load the data: x, tx, allx, graph
#     names = ['x', 'tx', 'allx', 'graph']
#     objects = []
#     for i in range(len(names)):
#         objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
#     x, tx, allx, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
#     test_idx_range = np.sort(test_idx_reorder)
#
#     if dataset == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#
#     return adj, features

def load_data(data_source):
    data = scipy.io.loadmat("data/{}_final.mat".format(data_source))
    labels = data["Label"]
    attributes = sp.csr_matrix(data["Attributes"])
    network = sp.lil_matrix(["Network"])

    return network, attributes, labels

def format_data(data_source):

    adj, features, labels = load_data(data_source)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # adj = adj_orig

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, adj_orig, num_features, num_nodes, features_nonzero, adj_norm, adj_label, features]
    feas = {}
    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]
        feas[retrieve_name(item)] = item

    return feas

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

