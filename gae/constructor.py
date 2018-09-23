import tensorflow as tf
from model import GCNModelAE, GCNModelVAE
from optimizer import OptimizerAE, OptimizerVAE
from preprocessing import construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder():
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)

    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    return model

def get_optimizer(model_str, model, placeholders, num_nodes, alpha):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds_attribute=model.attribute_reconstructions,
                          labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                          preds_structure=model.structure_reconstructions,
                          labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=alpha)

    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=placeholders['features'],
                           model=model, num_nodes=num_nodes)
    return opt

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # feed_dict.update({placeholders['dropout']: 0})
    # emb = sess.run(model.z_mean, feed_dict=feed_dict)

    _, reconstruct_loss, reconstruction_errors = sess.run([opt.opt_op, opt.cost, opt.reconstruction_errors], feed_dict=feed_dict)


    return reconstruction_errors, reconstruct_loss