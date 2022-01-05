import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

def fix_bbox(bbox: np.array) -> np.array:
    ''' Reformats the bounding box inputs into correct shape for TensorFlow display function '''
    temp = np.zeros_like(bbox)
    temp[0], temp[1] = bbox[1], bbox[0]
    temp[2], temp[3] = bbox[3], bbox[2]
    return temp

def data_augmentation(input: tf.Tensor, mask: tf.Tensor, bbox: tf.Tensor) -> tuple[tf.Tensor]:
    ''' Applies random flip or rotation to input and mask '''
    bbox_mask = np.copy(bbox)
    if np.random.rand() > 0.5:
        bbox_mask = create_mask(bbox, input.shape)
        if np.random.rand() > 0.5:
            input = tf.image.flip_left_right(input)
            mask = tf.image.flip_left_right(mask)
            bbox_mask = tf.image.flip_left_right(bbox_mask)
        else:
            input = tf.image.rot90(input)
            mask = tf.image.rot90(mask)
            bbox_mask = tf.image.rot90(bbox_mask)
        bbox_mask = get_bbox_from_mask(bbox_mask)
            
    return (input, mask, bbox_mask)


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
    box_img = tf.image.draw_bounding_boxes(tf.cast(tf.expand_dims(img, 0), tf.float32), fix_bbox(bbox_pred).reshape([1,1,4])/256, np.array([[0, 255, 0]]))
    
    ax1.imshow(tf.keras.utils.array_to_img(tf.squeeze(box_img_truth)))
    ax2.imshow(tf.keras.utils.array_to_img(tf.squeeze(box_img)))
    ax3.imshow(tf.keras.utils.array_to_img(mask_truth))
    ax4.imshow(tf.keras.utils.array_to_img(seg_max[0]))
    ax1.set_title('Truth')
    ax2.set_title('Prediction')
    ax3.set_title('Truth')
    ax4.set_title('Prediction')
    
def create_mask(bbox, input_shape):
    ''' Generates mask of bbox inputs '''
    bbox = bbox.astype(np.int32)
    shape = np.copy(input_shape)
    shape[-1] = 1
    temp = np.zeros(shape)
    for i in range(input_shape[0]):
        temp[i, bbox[i, 1]:bbox[i, 3], bbox[i, 0]:bbox[i, 2]] = 1
        
    return temp

def get_bbox_from_mask(mask):
    ''' Generates bbox from masks '''
    temp = np.zeros((mask.shape[0], 4))
    for i in range(mask.shape[0]):
        x, y = np.nonzero(mask[i])[:2]
        temp[i, 0] = np.min(y) # x min
        temp[i, 1] = np.min(x) # y min
        temp[i, 2] = np.max(y) # x max
        temp[i, 3] = np.max(x) # y max
        
    return temp

def calculate_iou(target_boxes, pred_boxes):
	''' Calculates intersection area union of bounding box prediction and truth - Code borrowed from https://github.com/AndrzejBandurski '''
	xA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])
	yA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])
	xB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])
	yB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])
	interArea = K.maximum(0.0, xB - xA) * K.maximum(0.0, yB - yA)
	boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
	boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
	iou = interArea / (boxAArea + boxBArea - interArea)
	return iou


def generator_img_baseline_data(images, masks):
    ''' Merges together datasets into a unified generator to pass for training '''
    a = images.as_numpy_iterator()
    b = masks.as_numpy_iterator()

    while True:
        X = a.next()
        Y = b.next()

        # Regularisation and shuffling
        X, Y = get_randomised_data([X, Y])
        yield X, Y