import tensorflow as tf
from keras.callbacks import CSVLogger
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from utils.helper_functions import get_mtl_training_log_path
from utils.loader import DataLoader
from models.effnet_encoder import EffnetEncoder
from models.mtl_framework import MTLFramework
from utils import tools, config

# Set configs
batch_size = 6
batch_size_val = 6
num_train, num_val, num_test = config.config['num_train'], config.config['num_val'], config.config['num_test']
img_height, img_width, channels = config.config['input_shape']

# Load our data pipeline
loader = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val)

# Train set
img_ds = loader.get_image_ds().repeat()
masks_ds = loader.get_mask_ds().repeat()
label_ds = loader.get_binary_ds().repeat()
bbox_ds = loader.get_bboxes_ds().repeat()

# Validation set
img_ds_val = loader.get_image_ds(val=True).repeat()
masks_ds_val = loader.get_mask_ds(val=True).repeat()
label_ds_val = loader.get_binary_ds(val=True).repeat()
bbox_ds_val = loader.get_bboxes_ds(val=True).repeat()

### CLEARS OLD MODELS IN CACHE
tf.keras.backend.clear_session()

# Get encoder
base_model_name = 'B0'
encoder = EffnetEncoder(base_model_name, (img_height, img_width, channels)).build_encoder(trainable=True)
encoder.summary()

# Use our MTL framework to custom build a model
mtl_builder = MTLFramework(encoder, (img_height, img_width, channels))
mtl_builder.add_segmentation_head()
mtl_builder.add_binary_classification_head(base_model_name, trainable=True)
mtl_builder.add_bbox_classification_head(base_model_name, trainable=True)
model = mtl_builder.build_mtl_model()
model.summary()


def generator_img():
    ''' Merges together datasets into a unified generator to pass for training '''
    # We use generators like this to abstract shuffling away from the loader and customise yielding
    # directly from the loaders like shuffling and augmentation.
    a = img_ds.as_numpy_iterator()
    b = masks_ds.as_numpy_iterator()
    c = label_ds.as_numpy_iterator()
    d = bbox_ds.as_numpy_iterator()

    while True:
        X = a.next()
        Y1 = b.next()
        Y2 = c.next()
        Y3 = d.next()

        # Regularisation and shuffling
        X, Y1, Y2, Y3 = tools.get_randomised_data([X, Y1, Y2, Y3])
        X, Y1, Y3 = tools.data_augmentation(X, Y1, Y3)

        yield X, (Y1, Y2, Y3)




def generator_img_val():
    ''' Merges together datasets into a unified generator to pass for training '''
    a = img_ds_val.as_numpy_iterator()
    b = masks_ds_val.as_numpy_iterator()
    c = label_ds_val.as_numpy_iterator()
    d = bbox_ds.as_numpy_iterator()

    while True:
        X = a.next()
        Y1 = b.next()
        Y2 = c.next()
        Y3 = d.next()

        yield X, (Y1, Y2, Y3)


# Initial train
model.compile(optimizer=keras.optimizers.Adam(),
              loss={'segnet_out': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    'bin_class_out': tf.keras.losses.BinaryCrossentropy(),
                    'bbox_out': tf.keras.losses.MeanAbsoluteError()},
              loss_weights=[1, 1, 1 / 100],  # Scale MAE to BC range
              metrics=['accuracy'])
history = model.fit(generator_img(), validation_data=generator_img_val(), epochs=15,
                    steps_per_epoch=num_train // batch_size, validation_steps=num_val // batch_size_val)
# Fine-tuning at lower learning rate
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss={'segnet_out': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    'bin_class_out': tf.keras.losses.BinaryCrossentropy(),
                    'bbox_out': tf.keras.losses.MeanAbsoluteError()},
              loss_weights=[1, 1, 1 / 100],  # Scale MAE to BC range
              metrics=['accuracy'])

csv_logger = CSVLogger(get_mtl_training_log_path(), separator=',', append=False)


history = model.fit(generator_img(), validation_data=generator_img_val(), epochs=10,
                    steps_per_epoch=num_train // batch_size, validation_steps=num_val // batch_size_val, callbacks=[csv_logger])
