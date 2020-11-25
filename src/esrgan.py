from os import stat
import tensorflow as tf
import tensorflow_hub as hub 

from exceptions import UnableToLoadModel
from base import SRModel


class ESRGAN(SRModel):
    SAVED_MODEL_PATH = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

    def load(self):
        if not self._loaded:
            try:
                self._model = hub.load(self.SAVED_MODEL_PATH)
            except Exception as ex:
                raise UnableToLoadModel(ex)
            else:
                self._loaded = True

    def target_size(self) -> tuple:
        return ()

    @staticmethod
    def preprocess_image(image: tf.Tensor) -> tf.Tensor:
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if image.shape[-1] == 4:
            image = image[...,:-1]
        
        # make image size divisible by 4
        hr_size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4

        hr_image = tf.image.crop_to_bounding_box(image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, axis=0)
