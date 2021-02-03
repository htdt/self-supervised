import tensorflow.compat.v2 as tf
from absl import flags

FLAGS = flags.FLAGS


class Whitening1D(tf.keras.layers.Layer):
    def __init__(self, eps=0, **kwargs):
        super(Whitening1D, self).__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        bs, c = x.shape
        x_t = tf.transpose(x, (1, 0))
        m = tf.reduce_mean(x_t, axis=1, keepdims=True)
        f = x_t - m
        ff_apr = tf.matmul(f, f, transpose_b=True) / (tf.cast(bs, tf.float32) - 1.0)
        ff_apr_shrinked = (1 - self.eps) * ff_apr + tf.eye(c) * self.eps
        sqrt = tf.linalg.cholesky(ff_apr_shrinked)
        inv_sqrt = tf.linalg.triangular_solve(sqrt, tf.eye(c))
        f_hat = tf.matmul(inv_sqrt, f)
        decorelated = tf.transpose(f_hat, (1, 0))
        return decorelated


def w_mse_loss(x):
    """ input x shape = (batch size * num_samples, proj_out_dim) """

    w = Whitening1D()
    num_samples = FLAGS.num_samples
    num_slice = num_samples * FLAGS.train_batch_size // (2 * FLAGS.proj_out_dim)
    x_split = tf.split(x, num_slice, 0)
    for i in range(num_slice):
        x_split[i] = w(x_split[i])
    x = tf.concat(x_split, 0)
    x = tf.math.l2_normalize(x, -1)

    x_split = tf.split(x, num_samples, 0)
    loss = 0
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            v = x_split[i] * x_split[j]
            loss += 2 - 2 * tf.reduce_mean(tf.reduce_sum(v, -1))
    loss /= num_samples * (num_samples - 1) // 2
    return loss
