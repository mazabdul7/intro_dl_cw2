# plot all metrics for the best model
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

from utils.config import config
import pandas as pd

from utils.helper_functions import get_unet_training_log_path, get_effNet_training_log_path, get_unet_model_path, \
    get_effNet_model_path, get_mtl_training_log_path
from utils.loader_cv import DataLoaderCV
from utils.tools import dice_binary

batch_size = config['unet_batch_size']
batch_size_val = config['unet_batch_size']
epochs = np.arange(0, 10)

# plot model1 vs model2 metrics
def plot_metrics_of_models(model_logs1, model1_name, model_logs2, model2_name, epochs, isMTL = False):
    plt.plot(epochs, model_logs1['loss'], label=f'{model1_name} loss')
    plt.plot(epochs, model_logs1['val_loss'], label=f'{model1_name}val_loss')

    plt.plot(epochs, model_logs2['loss'], label=f'{model2_name} loss')
    plt.plot(epochs, model_logs2['val_loss'], label=f'{model2_name} val_loss')
    plt.legend()
    plt.show()

    if isMTL:
        plt.plot(epochs, model_logs1['segnet_out_accuracy'], label=f'{model1_name} segnet_out_accuracy')
        plt.plot(epochs, model_logs1['val_segnet_out_accuracy'], label=f'{model1_name} val_segnet_out_accuracy')
    else:
        plt.plot(epochs, model_logs1['accuracy'], label=f'{model2_name} accuracy')
        plt.plot(epochs, model_logs1['val_accuracy'], label=f'{model2_name} val_accuracy')

    plt.plot(epochs, model_logs2['accuracy'], label=f'{model2_name} accuracy')
    plt.plot(epochs, model_logs2['val_accuracy'], label=f'{model2_name} val_accuracy')
    plt.legend()
    plt.show()

    if not isMTL:
        plt.plot(epochs, model_logs1['dice_binary'], label=f'{model1_name} dice_binary')
        plt.plot(epochs, model_logs1['val_dice_binary'], label=f'{model1_name} val_dice_binary')

        plt.plot(epochs, model_logs2['dice_binary'], label=f'{model2_name} dice_binary')
        plt.plot(epochs, model_logs2['val_dice_binary'], label=f'{model2_name} val_dice_binary')
        plt.legend()
        plt.show()


def compute_test_set_accuracy_for_model(loader, model):
    img_ds_test = loader.get_image_ds(test_mode=True)
    masks_ds_test = loader.get_mask_ds(test_mode=True)

    # Predict using best UNet model
    test_preds = model.predict(img_ds_test, batch_size=batch_size)
    formatted_test_preds = (test_preds >= 0.5).astype(np.uint8)

    # Compute and print test accuracy
    test_accuracy = np.sum(formatted_test_preds == masks_ds_test) / (masks_ds_test.shape[0] * (255 * 255))
    return test_accuracy


# Compare UNet to EffNet:
# we have found that the best model for UNet
# is model_3 (printed it after the cross val)
# and the best model for EffNet is also 3
# unet_logs =  pd.read_csv(get_unet_training_log_path(2), sep=',', engine='python')
# effnet_logs = pd.read_csv(get_effNet_training_log_path(2), sep=',', engine='python')
#
# plot_metrics_of_models(unet_logs, 'UNet', effnet_logs, 'EffNet' epochs)
#
# best_unet_model = load_model(get_unet_model_path(2), custom_objects={"dice_binary": dice_binary})
# unet_loader = DataLoaderCV(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=0, CV_iteration=0)
# unet_test_accuracy = compute_test_set_accuracy_for_model(unet_loader, best_unet_model)
# print(f"UNet Accuracy {unet_test_accuracy * 100}%")
#
# best_effnet_model = load_model(get_effNet_model_path(2), custom_objects={"dice_binary": dice_binary})
# effnet_loader = DataLoaderCV(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=0, CV_iteration=0)
# effnet_test_accuracy = compute_test_set_accuracy_for_model(effnet_loader, best_effnet_model)
# print(f"EffNet Accuracy {effnet_test_accuracy * 100}%")



# Compare MTL to EffNet:
# best model for EffNet is also 3
# best MTL model is the one in train_mtl
mtl_logs = pd.read_csv(get_mtl_training_log_path(), sep=',', engine='python')
effnet_logs = pd.read_csv(get_effNet_training_log_path(2), sep=',', engine='python')

plot_metrics_of_models(mtl_logs, 'MTL', effnet_logs, 'EffNet', epochs)