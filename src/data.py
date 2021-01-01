import tensorflow as tf
from os import path


IMAGES_DIR = './data/img'
CLEAR_IMAGES_DIR = path.join(IMAGES_DIR, 'clear')
DEGRADATED_IMAGES_DIR = path.join(IMAGES_DIR, 'degradated')
ANNOTATION_FILE_PATH = './data/annotation.json'
LR_IMAGE_SIZE = 128, 128
HR_IMAGE_SIZE = 256, 256

def read_image(fp, target_size):
    """
    Reads image and resizes it with Bicubic interpolation to the target size.
    """
    image_string = tf.io.read_file(fp)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return tf.image.resize(image, target_size, 'bicubic')
    

def prepare_example(hr_image_fp):
    """
    Reads high resolution image and makes one training example from it (low and high resolution images pair).
    """ 
    return read_image(hr_image_fp, LR_IMAGE_SIZE), read_image(hr_image_fp, HR_IMAGE_SIZE)


def get_dataset(images_dir):
    return tf.data.Dataset.list_files(path.join(images_dir, '*')).map(prepare_example)
