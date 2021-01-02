import matplotlib.gridspec as gridspec
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging


logging.basicConfig(filename='./out/logs.log', format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger()

def save_image(image, filename):
    """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save to.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(f'{filename}.jpg')
    print(f'Saved as {filename}.jpg')

def plot_multiple_images(batch, title="", cmap=None, figsize=None):
    """
    Plots batch of images from image tensors.
    """
    batch = tf.cast(tf.clip_by_value(batch, 0, 255), tf.uint8)
    
    X, y = batch
    
    N = X.shape[0]
    sqrtn = int(np.ceil(np.sqrt(N)))

    fig = plt.figure(figsize=figsize or (sqrtn, sqrtn))
    fig.suptitle(title)

    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(0, N, 2):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(X[i])
        
        ax = plt.subplot(gs[i+1])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(y[i])


def plot_single_image(image, title="", cmap=None):
    """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
    """
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.title(title)
