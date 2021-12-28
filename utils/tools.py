import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

def fix_bbox(bbox: np.array) -> np.array:
    ''' Reformats the bounding box inputs into correct shape for TensorFlow display function '''
    temp = np.zeros_like(bbox)
    temp[0], temp[1] = bbox[1], bbox[0]
    temp[2], temp[3] = bbox[3], bbox[2]
    return temp

def data_augmentation(input: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor]:
    ''' Applies random flip or rotation to input and mask '''
    if np.random.rand() > 0.5:
        if np.random.rand() > 0.5:
            input = tf.image.flip_left_right(input)
            mask = tf.image.flip_left_right(mask)
        else:
            input = tf.image.rot90(input)
            mask = tf.image.rot90(mask)

    return (input, mask)

def get_randomised_data(x_data: np.array, y_data_1: np.array, y_data_2: np.array) -> tuple[np.array]:
    ''' Performs consistent shuffling on input arrays '''
    dataset_size = len(x_data)
    dataset_indices = list(range(dataset_size))

    train_indices = random.sample(dataset_indices, dataset_size)

    x_train = x_data[train_indices, ...]
    y_1_train = y_data_1[train_indices, ...]
    y_2_train = y_data_2[train_indices, ...]

    return (x_train, y_1_train, y_2_train)

def show_seg_pred(img: np.array, mask_truth: np.array, mask_pred: np.array):
    ''' Show segmentation prediction '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,12))
    seg_max = tf.argmax(mask_pred, axis=-1)
    seg_max = seg_max[..., tf.newaxis][0]
    ax1.imshow(tf.keras.utils.array_to_img(tf.squeeze(img)))
    ax2.imshow(tf.keras.utils.array_to_img(mask_truth))
    ax3.imshow(seg_max)