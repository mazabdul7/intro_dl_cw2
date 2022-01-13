import numpy as np


def get_unet_model_path(fold):
    return f'model_weights/UNet/CV/best_unet_model_{fold+1}.h5'


def get_unet_training_log_path(fold):
    return f'model_weights/UNet/CV/unet_training_{fold+1}.log'


def get_mtl_training_log_path(val: bool):
    if val:
        return f'model_weights/MTL/CV/mtl_val_training.log'
    else:
        return f'model_weights/MTL/CV/mtl_training.log'


def get_effNet_model_path(fold):
    return f'model_weights/effNet/CV/best_EFFNET_{fold+1}.tf'


def get_effNet_training_log_path(fold):
    return f'model_weights/effNet/CV/EFFNET_training_{fold+1}.log'

def print_model_metric_analysis(history):
    print(f'\tTraining:')
    print(f'\t\tAccuracy: {np.mean(history["accuracy"]) * 100} %')
    print(f'\t\tLoss: {np.mean(history["loss"]) * 100} %')
    print(f'\t\tDice Binary: {np.mean(history["dice_binary"]) * 100} %\n')

    print(f'\tValidation:')
    print(f'\t\tAccuracy: {np.mean(history["val_accuracy"]) * 100} %')
    print(f'\t\tLoss: {np.mean(history["val_loss"]) * 100} %')
    print(f'\t\tDice Binary: {np.mean(history["val_dice_binary"]) * 100} %\n')


def get_metric_percentage_for_CV(model_histories, cross_validation_folds, metric):
    return np.mean([model_histories[i][metric] for i in range(cross_validation_folds)])*100


def print_models_average_metric_analysis(model_histories, cross_validation_folds):
    print(f'\tTraining:')
    print(f'\t\tAccuracy: {get_metric_percentage_for_CV(model_histories, cross_validation_folds, "accuracy")} %')
    print(f'\t\tLoss: {get_metric_percentage_for_CV(model_histories, cross_validation_folds, "loss")} %')
    print(
        f'\t\tDice Binary: {get_metric_percentage_for_CV(model_histories, cross_validation_folds, "dic_binary")} %\n')

    print(f'\tValidation:')
    print(f'\t\tAccuracy: {get_metric_percentage_for_CV(model_histories, cross_validation_folds, "val_accuracy")} %')
    print(f'\t\tLoss: {get_metric_percentage_for_CV(model_histories, cross_validation_folds, "val_loss")} %')
    print(
        f'\t\tDice Binary: {get_metric_percentage_for_CV(model_histories, cross_validation_folds, "val_dic_binary")} %\n')