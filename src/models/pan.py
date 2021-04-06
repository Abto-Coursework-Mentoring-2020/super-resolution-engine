import tensorflow as tf

from .exceptions import UnableToLoadModel
from .base import SRModel
from ._models.pan import PixelAttentionSRNetwork


class PAN(SRModel):
    def __init__(self, scale:int, input_lr_shape:tuple, checkpoints_path:str, checkpoint_name='pix-att-net-ckpt'):
        self._scale = scale
        self._input_lr_shape = input_lr_shape
        self._checkpoints_path = checkpoints_path
        self._checkpoint_name = checkpoint_name

    def load(self):
        if self._loaded:
            return
        try:
            self._model = PixelAttentionSRNetwork(
                self._input_lr_shape, feat_extr_n_filters=30, upsamp_n_filters=20, n_blocks=16, scale=self._scale
            )

            checkpoint = tf.train.Checkpoint(sr_net=self._model)

            self._model.build(self._input_lr_shape)
            ckpt_manager = tf.train.CheckpointManager(checkpoint, self._checkpoints_path, 1, checkpoint_name=self._checkpoint_name)

            # try restoring previous checkpoint or initialize a new one
            restored_ckpt_path = ckpt_manager.restore_or_initialize()
            if restored_ckpt_path is not None:
                print(f'Restored model state from checkpoint {repr(restored_ckpt_path)}.')
            else:
                print(f'Unable to find latest checkpoint {repr(self._checkpoint_name)} under {repr(self._checkpoints_path)}.')
        except Exception as ex:
            raise UnableToLoadModel(ex)
        else:
            self._loaded = True

    @staticmethod
    def preprocess_images(images: tf.Tensor) -> tf.Tensor:
        return tf.cast(images, tf.float32) / 255.0

    @staticmethod
    def deprocess_images(images: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.clip_by_value(images*255.0, 0.0, 255.0), tf.uint8)
