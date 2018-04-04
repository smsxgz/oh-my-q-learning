import numpy as np
import tensorflow as tf


def intprod(shape):
    return int(np.prod(shape))


def flat(var_list):
    return tf.concat(axis=0, values=[tf.reshape(v, [-1]) for v in var_list])


def unflat(flat_tensor, shapes):
    output = []
    start = 0
    for shape in shapes:
        size = intprod(shape)
        output.append(tf.reshape(flat_tensor[start:start + size], shape))
        start += size
    return output


class Order2Optimizer(object):
    def __init__(self, parameters, lr, alpha=None):
        self.lr = lr
        self.alpha = alpha
        self.parameters = parameters
        self.shapes = [v.shape for v in parameters]
        self.shape = sum(intprod(shape) for shape in self.shapes)

        with tf.name_scope('optimizer'):
            self.pre_flat_parameters = tf.Variable(
                tf.truncated_normal([self.shape], stddev=0.01),
                trainable=False)
            self.pre_flat_grads = tf.Variable(
                tf.truncated_normal([self.shape], stddev=0.01),
                trainable=False)

    def minimize(self, loss, init_method=None):
        parameters = self.parameters
        pre_flat_parameters = self.pre_flat_parameters
        pre_flat_grads = self.pre_flat_grads

        global_step = tf.train.get_global_step()
        update_step_op = global_step.assign_add(1)

        flat_parameters = flat(parameters)
        grads = tf.gradients(loss, parameters)
        flat_grads = flat(grads)

        assign_op = [
            pre_flat_parameters.assign(flat_parameters),
            pre_flat_grads.assign(flat_grads)
        ]

        # Get previous parameters and grads' init_op
        if init_method == 'reinit':
            with tf.control_dependencies(assign_op):
                reinit_op = []
                for var in parameters:
                    reinit_op.append(var.initializer)
                return tf.group(*reinit_op, update_step_op)

        elif init_method == 'sgd':
            with tf.control_dependencies(assign_op):
                train_op = []
                for var, g in zip(parameters, grads):
                    train_op.append(var.assign_add(-self.lr * g))
                return tf.group(*train_op, update_step_op)

        elif init_method == 'random':
            return update_step_op

        # Get train_op
        elif init_method is None:
            if self.alpha is None:
                frac = tf.reduce_sum(tf.square(pre_flat_grads - flat_grads))
                alpha1 = tf.reduce_sum(
                    (pre_flat_grads - flat_grads) * pre_flat_grads) / frac
                beta1 = alpha1
            else:
                alpha1, beta1 = self.alpha

            tf.identity(alpha1, name='alpha1')
            tf.summary.scalar('alpha1', alpha1)

            new_parameters = alpha1 * flat_parameters + (
                1 - alpha1) * pre_flat_parameters
            new_parameters = new_parameters - self.lr * (
                beta1 * flat_grads + (1 - beta1) * pre_flat_grads)
            new_parameters = unflat(new_parameters, self.shapes)

            with tf.control_dependencies(new_parameters):
                with tf.control_dependencies(assign_op):
                    train_op = []
                    for var, new_var in zip(parameters, new_parameters):
                        train_op.append(var.assign(new_var))
                    return tf.group(*train_op, update_step_op)
        else:
            raise Exception('Illegal init method...')
