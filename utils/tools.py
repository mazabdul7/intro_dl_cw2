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

def get_randomised_data(args) -> tuple[np.array]:
    ''' Performs consistent shuffling on input arrays '''
    dataset_size = len(args[0])
    dataset_indices = list(range(dataset_size))

    train_indices = random.sample(dataset_indices, dataset_size)

    return (ds[train_indices, ...] for ds in args)

def show_seg_pred(img: np.array, mask_truth: np.array, mask_pred: np.array, bbox_truth: np.array, bbox_pred: np.array):
    ''' Show segmentation prediction with bounding box pred '''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,12))
    seg_max = tf.where(mask_pred > 0, 1, 0)
    box_img_truth = tf.image.draw_bounding_boxes(tf.cast(tf.expand_dims(img, 0), tf.float32), fix_bbox(bbox_truth).reshape([1,1,4])/256, np.array([[255, 0, 0]]))
    box_img = tf.image.draw_bounding_boxes(tf.cast(tf.expand_dims(img, 0), tf.float32), fix_bbox(bbox_pred).reshape([1,1,4])/256, np.array([[255, 0, 0]]))
    
    ax1.imshow(tf.keras.utils.array_to_img(tf.squeeze(box_img_truth)))
    ax2.imshow(tf.keras.utils.array_to_img(tf.squeeze(box_img)))
    ax3.imshow(tf.keras.utils.array_to_img(mask_truth))
    ax4.imshow(tf.keras.utils.array_to_img(seg_max[0]))