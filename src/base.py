import tensorflow as tf
from abc import ABC, abstractproperty, abstractmethod, abstractstaticmethod, abstractstaticmethod

from exceptions import ModelIsNotLoaded


class SRModel(ABC):
    _loaded = False
    _model = None
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

    @abstractproperty
    def target_size(self) -> tuple:
        """
        Target image size of the SR model.
        """
        pass

    @abstractstaticmethod
    def preprocess_image(image: tf.Tensor) -> tf.Tensor:
        """ 
        Preprocesseses an image to make it model ready.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Loads model's weights.
        """
        pass

    def enhance(self, images: tf.Tensor) -> tf.Tensor:
        """
        Runs already prepared images through the model to enchance them.er n

        Args:
            images (tf.Tensor): [description]

        Returns:
            (tf.Tensor): enchanced images
        """
        if not self.is_loaded():
            raise ModelIsNotLoaded
        
        return self._model(images)
    
    def __call__(self, images: tf.Tensor) -> tf.Tensor:
        return self.enhance(images)
