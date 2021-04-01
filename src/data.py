import cv2
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
    return tf.constant(cv2.resize(cv2.imread(fp.numpy().decode('utf-8')), target_size), dtype=tf.float32)
    

def prepare_example(hr_image_fp, ret_img_name=False, lr_img_size=(128, 128), hr_img_size=(256, 256)):
    """
    Reads high resolution image and makes one training example from it (low and high resolution images pair).
    """ 
    example = read_image(hr_image_fp, lr_img_size), read_image(hr_image_fp, hr_img_size)
    if ret_img_name:
        img_name = path.split(str(hr_image_fp))[1].split('.')[0]
        return example + (tf.constant(img_name),)
    else:    
        return example


def get_dataset(images_dir):
    return tf.data.Dataset.list_files(path.join(images_dir, '*')).map(lambda fp: tf.py_function(prepare_example, [fp], (tf.float32, tf.float32)))

