import tensorflow as tf


@tf.function
def l1_image_loss(batch_true, batch_pred):
    return tf.reduce_sum(tf.abs(batch_true - batch_pred)) / tf.cast(tf.reduce_prod(batch_true.shape), tf.float32)
