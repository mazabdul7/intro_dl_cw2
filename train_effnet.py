import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import random

from models.unet_model import UNet
from utils.config import config
from utils.loader_cv import DataLoaderCV
from utils.loader import DataLoader
from utils.tools import generator_img_baseline_data, dice_binary
from utils.helper_functions import  print_model_metric_analysis, \
print_models_average_metric_analysis, get_effnet_model_path, get_effnet_training_log_path

from models.effnet_encoder import EffnetEncoder
from models.mtl_framework import MTLFramework

# Set configs
batch_size = config['batch_size']
batch_size_val = config['batch_size']
num_train = config['num_train']
num_val = config['num_val']
cross_validation_folds = config['cross_validation_folds']

num_epochs = 20

# Instantiate and build model
input_shape = config['input_shape']
base_model_name = 'B0'
encoder = EffnetEncoder(base_model_name, input_shape).build_encoder(trainable=True)

# Using the MTL framework to custom build an EffNet model
effnet_builder = MTLFramework(encoder, input_shape)
effnet_builder.add_segmentation_head()
model = effnet_builder.build_mtl_model()

# Model histories is a list
# including the history of the best EffNet model per fold
model_histories = []
for fold in range(cross_validation_folds):
      print(f"-------------------- start cross val {fold+1} --------------------")
      train_loader = DataLoaderCV(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=1, CV_iteration=fold, fold=cross_validation_folds)
      val_loader = DataLoaderCV(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=2, CV_iteration=fold, fold=cross_validation_folds)

      # Get training data
      train_imgs = train_loader.get_image_ds().repeat()
      train_masks = train_loader.get_mask_ds().repeat()

      # Get validation data
      val_imgs = val_loader.get_image_ds().repeat()
      val_masks = val_loader.get_mask_ds().repeat()

      # Clears old models from the cache
      tf.keras.backend.clear_session()

      model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[{'accuracy': 'accuracy', 'dice_binary': dice_binary}])
      model.summary()

      # Checkpointer that saves the best effnet model
      # and a cvs_logger that saves the history of each effnet model
      checkpointer = ModelCheckpoint(get_effnet_model_path(fold), verbose=1, save_best_only=True)
      csv_logger = CSVLogger(get_effnet_training_log_path(fold), separator=',', append=False)

      # Fit training and validation data
      history = model.fit(generator_img_baseline_data(train_imgs, train_masks), validation_data=generator_img_baseline_data(val_imgs, val_masks), batch_size=batch_size, epochs=num_epochs, callbacks=[checkpointer, csv_logger], steps_per_epoch=num_train//batch_size, validation_steps=num_val//batch_size_val)

      model_histories.append(history.history)

# Printing the metric analysis per fold and average
print('Scores per fold:\n')
for i in range(cross_validation_folds):
    history = model_histories[i]
    print(f'For fold {i+1}:')
    print_model_metric_analysis(history)

print('Average Scores for EffNet:')
print_models_average_metric_analysis(model_histories, cross_validation_folds)

# Pick best model out of the 3 - used loss we can use some other factor
average_loss_per_model = [np.mean(model_histories[i]["loss"]) for i in range(cross_validation_folds)]
best_model_idx = np.argmin(average_loss_per_model)
print(f'Best EffNet model: {best_model_idx+1}')

best_model_history = model_histories[best_model_idx]
epochs = np.arange(0, num_epochs)

# plot all metrics for the best model
plt.plot(epochs, best_model_history['loss'], label='loss')
plt.plot(epochs, best_model_history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(epochs, best_model_history['accuracy'], label='accuracy')
plt.plot(epochs, best_model_history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

plt.plot(epochs, best_model_history['dice_binary'], label='dice_binary')
plt.plot(epochs, best_model_history['val_dice_binary'], label='val_dice_binary')
plt.legend()
plt.show()


# load best model
best_model = load_model(get_effnet_model_path(best_model_idx), custom_objects={"dice_binary": dice_binary})

# Load and get test data
test_loader = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val)

img_ds_test = test_loader.get_image_ds(test_mode=True)
masks_ds_test = test_loader.get_mask_ds(test_mode=True)

# Predict using best UNet model
test_preds = best_model.predict(img_ds_test, batch_size=batch_size)
formatted_test_preds = (test_preds >= 0.5).astype(np.uint8)

# Compute and print test accuracy
test_accuracy = np.sum(formatted_test_preds == masks_ds_test) / (masks_ds_test.shape[0] * (255 * 255))
print(f"Accuracy {test_accuracy * 100}%")


# Plot 3 ground truth/prediction masks
idx = list(range(img_ds_test.shape[0]))
random.shuffle(idx)
for i in range(3):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(tf.keras.utils.array_to_img( masks_ds_test[idx[i]]))
    ax1.set_title('Truth')
    ax2.imshow(tf.keras.utils.array_to_img(formatted_test_preds[idx[i]]))
    ax2.set_title('Prediction')

    plt.show()

















