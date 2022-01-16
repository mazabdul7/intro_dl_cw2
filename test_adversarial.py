import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import config

# Set configs
batch_size = config.config['batch_size']
batch_size_val = config.config['batch_size']
num_train, num_val, num_test = config.config['num_train'], config.config['num_val'], config.config['num_test']
img_height, img_width, channels = config.config['input_shape']

print('Loading adversarial data...')
# Load adversarial inputs
train_ds = keras.preprocessing.image.DirectoryIterator(
    r'adversarial_imgs/data', tf.keras.preprocessing.image.ImageDataGenerator(), target_size=(img_height, img_width), batch_size=1, seed=777)
train_masks = keras.preprocessing.image.DirectoryIterator(
    r'adversarial_imgs/masks', tf.keras.preprocessing.image.ImageDataGenerator(), target_size=(img_height, img_width), batch_size=1, seed=777, color_mode='grayscale')

print('Loading normal MTL model...')
# Load model to test on adversarial inputs (Normal MTL)
model = tf.keras.models.load_model('model_weights/EffishingNetN')

print('Performing model performance tests...')
# Predict on test-set
seg_pred, bin_pred, bbox_pred = model.predict(train_ds, batch_size=10)
seg_pred = tf.where(seg_pred >= 0, 1, 0) # Convert to {0,1} binary classes
c2 = 0
for i in range(120):
    c2 += np.sum(seg_pred[i] == train_masks[i][0])
print(f'Seg Accuracy: {round(c2*100/((seg_pred.shape[0]*(img_height*img_width))), 3)}%')

print('Loading attention MTL model...')
# Load model to test on adversarial inputs (Attention MTL)
model = tf.keras.models.load_model('model_weights/EffishingNetAtt_Eff')

print('Performing model performance tests...')
# Predict on test-set
seg_pred, bin_pred, bbox_pred = model.predict(train_ds, batch_size=10)
seg_pred = tf.where(seg_pred >= 0, 1, 0) # Convert to {0,1} binary classes
c2 = 0

for i in range(120):
    c2 += np.sum(seg_pred[i] == train_masks[i][0])
print(f'Seg Accuracy: {round(c2*100/((seg_pred.shape[0]*(img_height*img_width))), 3)}%')


