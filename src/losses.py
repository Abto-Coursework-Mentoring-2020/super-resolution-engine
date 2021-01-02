import tensorflow as tf


@tf.function
def l1_image_loss(batch_true, batch_pred):
    return tf.reduce_sum(tf.abs(batch_true - batch_pred)) / tf.cast(tf.reduce_prod(batch_true.shape), tf.float32)


@tf.function
def multiscale_ssim_loss(batch_true, batch_pred, max_val=tf.constant(255.0)):
    return tf.reduce_mean(1 - tf.image.ssim_multiscale(batch_true, batch_pred, max_val))
