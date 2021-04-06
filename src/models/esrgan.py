from os import stat
import tensorflow as tf
import tensorflow_hub as hub 

from .exceptions import UnableToLoadModel
from .base import SRModel


class ESRGAN(SRModel):
    SAVED_MODEL_PATH = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

    def load(self):
        if self._loaded:
            return
        try:
            self._model = hub.load(self.SAVED_MODEL_PATH)
        except Exception as ex:
            raise UnableToLoadModel(ex)
        else:
            self._loaded = True

    def scale(self) -> int:
        return -1

    @staticmethod
    def preprocess_images(images: tf.Tensor) -> tf.Tensor:
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if images.shape[-1] == 5:
            images = images[...,:-1]
        return tf.cast(images, tf.float32)

    @staticmethod
    def deprocess_images(images: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.clip_by_value(images, 0.0, 255.0), tf.uint8)
