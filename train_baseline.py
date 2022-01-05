import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from PIL import Image
import matplotlib.pyplot as plt
import random

from models.unet_model import UNet
from utils.loader import DataLoader
from utils.tools import generator_img_baseline_data

img_w = 256
img_h = 256
img_channels = 3
mask_channels = 1

batch_size = 16
batch_size_val = 16

num_train = 2210
num_val = 738

loader = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val)

train_imgs = loader.get_image_ds().repeat()
train_masks = loader.get_mask_ds().repeat()

val_imgs = loader.get_image_ds(val=True).repeat()
val_masks = loader.get_mask_ds(val=True).repeat()

training_data = generator_img_baseline_data(train_imgs, train_masks)
val_data = generator_img_baseline_data(val_imgs, val_masks)

input_shape = (img_h, img_w, img_channels)
model = UNet(input_shape=input_shape).build_model()

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint('best-unet-model.h5', verbose=1, save_best_only=True)
results = model.fit(generator_img_baseline_data(train_imgs, train_masks), validation_data=generator_img_baseline_data(val_imgs, val_masks), batch_size=batch_size, epochs=50, callbacks=[checkpointer], steps_per_epoch=num_train//batch_size, validation_steps=num_val//batch_size_val)

model.save('saved_UNet_model')
model = load_model('best-unet-model.h5')