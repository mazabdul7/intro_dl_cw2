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
from utils.config import config
from utils.loader import DataLoader
from utils.tools import generator_img_baseline_data


batch_size = 16
batch_size_val = 16

num_train = config['num_train']
num_val = config['num_val']

loader = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val)

# get training data
train_imgs = loader.get_image_ds().repeat()
train_masks = loader.get_mask_ds().repeat()

# get validation data
val_imgs = loader.get_image_ds(val=True).repeat()
val_masks = loader.get_mask_ds(val=True).repeat()

# Instantiate and build model
input_shape = config['input_shape']
model = UNet(input_shape=input_shape).build_model()

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()

# fit training and validation data
# we have a checkpointer that saves the best unet model
checkpointer = ModelCheckpoint('model_weights/best_UNet_model.h5', verbose=1, save_best_only=True)
results = model.fit(generator_img_baseline_data(train_imgs, train_masks), validation_data=generator_img_baseline_data(val_imgs, val_masks), batch_size=batch_size, epochs=50, callbacks=[checkpointer], steps_per_epoch=num_train//batch_size, validation_steps=num_val//batch_size_val)

# load best model
model = load_model('model_weights/best_UNet_model.h5')

# get test data
img_ds_test = loader.get_image_ds(test_mode=True)
masks_ds_test = loader.get_mask_ds(test_mode=True)

test_preds = model.predict(img_ds_test, batch_size=batch_size)
formatted_test_preds = (test_preds >= 0.5).astype(np.uint8)

seg_acc = np.sum(formatted_test_preds == masks_ds_test)/(masks_ds_test.shape[0]*(255*255))
print(f"Accuracy {seg_acc*100}%")

# print 3 ground truth/prediction masks
idx = list(range(img_ds_test.shape[0]))
random.shuffle(idx)
for i in range(3):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(tf.keras.utils.array_to_img( masks_ds_test[idx[i]]))
    ax1.set_title('Truth')
    ax2.imshow(tf.keras.utils.array_to_img(formatted_test_preds[idx[i]]))
    ax2.set_title('Prediction')

    plt.show()

