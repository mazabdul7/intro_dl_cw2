import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import matplotlib.pyplot as plt
import random

from models.unet_model import UNet
from utils.config import config
from utils.loader import DataLoader
from utils.tools import generator_img_baseline_data, dice_binary

unet_model_path = 'model_weights/UNet/CV'
batch_size = config['unet_batch_size']
batch_size_val = config['unet_batch_size']

num_train = config['num_train']
num_val = config['num_val']

cross_validation_folds = config['cross_validation_folds']

num_epochs = 10

# Instantiate and build model
input_shape = config['input_shape']
model = UNet(input_shape=input_shape).build_model()


def get_mean_metric(metric, model_histories):
    return np.mean(model_histories[i][metric] for i in range(cross_validation_folds)) * 100

model_histories = []
for fold in range(cross_validation_folds):
    print(f"-------------------- start cross val {fold + 1} --------------------")
    # CrossVal: 0 = no CV, 1 = training set, 2 = val set
    loader1 = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=1, CV_iteration=fold)
    loader2 = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=2, CV_iteration=fold)

    # get training data
    train_imgs = loader1.get_image_ds().repeat()
    train_masks = loader1.get_mask_ds().repeat()

    # get validation data
    val_imgs = loader2.get_image_ds().repeat()
    val_masks = loader2.get_mask_ds().repeat()

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[{'accuracy': 'accuracy', 'dice_binary': dice_binary}])
    model.summary()

    # fit training and validation data
    # we have a checkpointer that saves the best unet model
    checkpointer = ModelCheckpoint(f'{unet_model_path}/best_UNet_model_{fold+1}.h5', verbose=1, save_best_only=True)
    history = model.fit(generator_img_baseline_data(train_imgs, train_masks), validation_data=generator_img_baseline_data(val_imgs, val_masks), batch_size=batch_size, epochs=num_epochs, callbacks=[checkpointer], steps_per_epoch=num_train//batch_size, validation_steps=num_val//batch_size_val)

    model_histories.append(history.history)

print('Scores per fold:\n')
for i in range(cross_validation_folds):
    history = model_histories[i]
    print(f'For fold {i+1}:')

    print(f'\tTraining:')
    print(f'\t\tAccuracy: {np.mean(history["accuracy"])*100} %')
    print(f'\t\tLoss: {np.mean(history["loss"])*100} %')
    print(f'\t\tDice Binary: {np.mean(history["dice_binary"])*100} %\n')

    print(f'\tValidation:')
    print(f'\t\tAccuracy: {np.mean(history["val_accuracy"])*100} %')
    print(f'\t\tLoss: {np.mean(history["val_loss"])*100} %')
    print(f'\t\tDice Binary: {np.mean(history["val_dice_binary"])*100} %\n')

print('Average Scores for UNET:')
print(f'\tTraining:')
print(f'\t\tAccuracy: {get_mean_metric("accuracy", model_histories)} %')
print(f'\t\tLoss: {get_mean_metric("loss", model_histories)} %')
print(f'\t\tDice Binary: {get_mean_metric("dice_binary", model_histories)} %\n')

print(f'\tValidation:')
print(f'\t\tAccuracy: {get_mean_metric("val_accuracy", model_histories)} %')
print(f'\t\tLoss: {get_mean_metric("val_loss", model_histories)} %')
print(f'\t\tDice Binary: {get_mean_metric("val_dice_binary", model_histories)} %\n')

# pick best model out of the 3 - used loss we can use some other factor
best_model_idx = np.argmin(np.mean(model_histories[i]["loss"]) for i in range(cross_validation_folds))

best_model_history = model_histories[best_model_idx]
epochs = np.arange(0, num_epochs)
# plot all metrics
plt.plot(epochs, best_model_history.history['loss'], label='loss')
plt.plot(epochs, best_model_history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(epochs, best_model_history.history['accuracy'], label='accuracy')
plt.plot(epochs, best_model_history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

plt.plot(epochs, best_model_history.history['dice_binary'], label='dice_binary')
plt.plot(epochs, best_model_history.history['val_dice_binary'], label='val_dice_binary')
plt.legend()
plt.show()

# load best model
best_model = load_model(f'{unet_model_path}/best_UNet_model_{best_model_idx}.h5', custom_objects={"dice_binary": dice_binary})

# fixme not sure if this is right
loader = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=0, CV_iteration=0)

# get test data
img_ds_test = loader.get_image_ds(test_mode=True)
masks_ds_test = loader.get_mask_ds(test_mode=True)

test_preds = best_model.predict(img_ds_test, batch_size=batch_size)
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


