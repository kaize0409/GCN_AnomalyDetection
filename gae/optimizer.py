import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds_attribute, labels_attribute, preds_structure, labels_structure, alpha):

        # attribute reconstruction loss
        diff_attribute = tf.square(preds_attribute - labels_attribute)
        self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_attribute, 1))
        # self.reconstruction_errors =  tf.losses.mean_squared_error(labels= labels, predictions=preds)
        self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)

        # structure reconstruction loss
        diff_structure = tf.square(preds_structure - labels_structure)

        self.structure_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_structure, 1))
        self.structure_cost = tf.reduce_mean(self.structure_reconstruction_errors)


        self.reconstruction_errors = tf.multiply(alpha, self.attribute_reconstruction_errors) + tf.multiply(1-alpha, self.structure_reconstruction_errors)
        self.cost = alpha * self.attribute_cost + (1-alpha) * self.structure_cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        # self.grads_vars = self.optimizer.compute_gradients(self.cost)



class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
