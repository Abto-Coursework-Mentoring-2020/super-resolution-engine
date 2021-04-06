import tensorflow as tf
from abc import ABC, abstractmethod, abstractstaticmethod

from .exceptions import ModelIsNotLoaded


class SRModel(ABC):
    _loaded = False
    _model = None
    _scale = 1
    """
    Base Super Resolution model that main interface methods.
    """

    def is_loaded(self) -> bool:
        """
        Returns whether the model is loaded or not.

        Returns:
            (bool): condition state
        """

        return self._loaded

    @property
    def scale(self) -> int:
        """
        Factor of the input image enhancement.
        """
        return self._scale

    @abstractstaticmethod
    def preprocess_images(images: tf.Tensor) -> tf.Tensor:
        """ 
        Preprocesseses an image to make it model-ready.
        """
        pass

    @abstractstaticmethod
    def deprocess_images(images: tf.Tensor) -> tf.Tensor:
        """ 
        Inverts any processing done in `preprocess_images`.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Loads model's weights.
        """
        pass

    def enhance(self, images: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Runs already prepared images through the model to enchance them.

        Args:
            images (tf.Tensor): input images tensor of shape (batch_size, H, W, C)

        Returns:
            (tf.Tensor): enchanced images
        """
        if not self.is_loaded():
            raise ModelIsNotLoaded
        
        return self.deprocess_images(self._model(self.preprocess_images(images), *args, **kwargs))

    def __call__(self, images: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        return self.enhance(images, *args, **kwargs)
