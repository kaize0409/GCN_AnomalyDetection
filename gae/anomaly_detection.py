from __future__ import division
from __future__ import print_function
import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from constructor import get_placeholder, get_model, get_optimizer, update
import numpy as np
from input_data import format_data
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


class AnomalyDetectionRunner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']


    def erun(self):
        model_str = self.model
        # load data
        feas = format_data(self.data_name)
        print("feature number: {}".format(feas['num_features']))

        # Define placeholders
        placeholders = get_placeholder()

        # construct model
        gcn_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, gcn_model, placeholders, feas['num_nodes'], FLAGS.alpha)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(1, self.iteration+1):

            reconstruction_errors, reconstruction_loss = update(gcn_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(reconstruction_loss))

            if epoch % 100 == 0:
                y_true = [label[0] for label in feas['labels']]
                auc = roc_auc_score(y_true, reconstruction_errors)
                print(auc)

        sorted_errors = np.argsort(-reconstruction_errors, axis=0)
        with open('output/{}-ranking.txt'.format(self.data_name), 'w') as f:
            for index in sorted_errors:
                f.write("%s\n" % feas['labels'][index][0])

	df = pd.DataFrame({'AD-GCA':reconstruction_errors})
	df.to_csv('output/{}-scores.csv'.format(self.data_name), index=False, sep=',')
	




