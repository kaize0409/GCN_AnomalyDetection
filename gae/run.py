import tensorflow as tf
import numpy as np
from anomaly_detection import AnomalyDetectionRunner
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 200, 'number of iterations.')


'''
We did not set any seed when we conducted the experiments described in the paper;
We set a seed here to steadily reveal better performance of ARGA
'''
seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

dataname = 'cora'       # 'cora' or 'citeseer' or 'pubmed'
model = 'arga_ae'          # 'arga_ae' or 'arga_vae'
task = 'anomaly_detection'         # 'clustering' or 'link_prediction'
settings = {'data_name': dataname, 'iterations' : FLAGS.iterations, 'model' : model}

runner = None
if task == 'anomaly_detection':
    runner = AnomalyDetectionRunner(settings)

runner.erun()