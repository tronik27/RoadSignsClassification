import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
# from tensorflow_addons.metrics import F1Score
import matplotlib.pyplot as plt
from DataPreprocessing import CustomDataGen
from ResNet import ResNetModel
import numpy as np
import seaborn as sns
import datetime
import shutil
import os
from typing import Tuple


class RoadSignsClassification:
    def __init__(self, batch_size: int, target_size: Tuple[int, int, int], metric_names: list, num_classes: int,
                 input_name: str = 'input', output_name: str = 'output', regularization: float = 0.0005,
                 num_filters: int = 16, learning_rate: float = 0.001,
                 path_to_model_weights: str = 'road_signs_model/weights') -> None:
        self.metric_names = metric_names
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.path_to_model_weights = path_to_model_weights
        self.nn = ResNetModel(input_shape=self.target_size, num_classes=num_classes, num_filters=num_filters,
                              regularization=regularization, input_name=input_name, output_name=output_name).build()
        self.nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=self.get_metrics())
        self.model_summary = self.nn.summary()

    def train(self,
              images: str,
              labels: str,
              validation_images: str = '',
              validation_labels: str = '',
              epochs: int = 100,
              apply_sample_weight: bool = True,
              augmentation: dict = {},
              show_learning_curves: bool = False,
              min_delta: float = 0.0001,
              scheduler=None) -> None:

        train_datagen = CustomDataGen(meta_path=labels, path_to_image=images, batch_size=self.batch_size,
                                      target_size=self.target_size, train_mode=True, aug_config=augmentation,
                                      apply_weights=apply_sample_weight)
        validation_datagen = CustomDataGen(meta_path=validation_labels, path_to_image=validation_images,
                                           batch_size=self.batch_size, target_size=self.target_size,
                                           train_mode=False, aug_config=augmentation,
                                           apply_weights=apply_sample_weight)

        history = self.nn.fit(
            train_datagen,
            validation_data=validation_datagen,
            steps_per_epoch=train_datagen.number_of_images // self.batch_size,
            callbacks=self.get_callbacks(min_delta=min_delta, scheduler=scheduler),
            epochs=epochs
        )

        if show_learning_curves:
            self.plot_learning_curves(history)

    def evaluate(self, test_images: str, test_labels: str,  plot_conf_matrix=False):

        test_datagen = CustomDataGen(
            meta_path=test_labels,
            path_to_image=test_images,
            batch_size=self.batch_size,
            target_size=self.target_size,
            train_mode=False
        )
        self.nn.load_weights(self.path_to_model_weights)
        results = self.nn.evaluate(test_datagen, batch_size=self.batch_size)
        for i, metric in enumerate(self.nn.metrics_names):
            print('{}: {}'.format(metric, results[i]))
        # if plot_conf_matrix:
        #     self.plot_confusion_matrix(results)

    def get_callbacks(self, min_delta, scheduler=None):
        callbacks = []
        if scheduler:
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
            callbacks.append(lr_scheduler)

        tensorboard_callback = self.make_tensorboard()
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.0001,
                                                         factor=0.1, patience=3, min_lr=0.00001)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.path_to_model_weights, save_weights_only=True,
                                                        save_best_only=True, monitor='val_loss', mode='min')
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=min_delta)
        callbacks += [tensorboard_callback, reduce_lr, checkpoint, stop]
        return callbacks

    def get_metrics(self):
        # metrics = {'accuracy': CategoricalAccuracy(), 'precision': Precision(),
        #            'recall': Recall(), 'f1': F1Score(num_classes=self.num_classes, threshold=0.5)}
        metrics = {'accuracy': CategoricalAccuracy(), 'precision': Precision(),
                   'recall': Recall()}
        metric_list = list()
        for mertic in self.metric_names:
            metric_list.append(metrics[mertic])
        return metric_list

    def plot_learning_curves(self, metric_data) -> None:
        figure, axes = plt.subplots(len(metric_data), 1, figsize=(7, 5))
        for axe, metric in zip(axes, self.metric_names):
            name = metric.replace("_", " ").capitalize()
            axe.plot(metric_data.epoch, metric_data.history[metric], label='Train')
            axe.plot(metric_data.epoch, metric_data.history['val_' + metric], linestyle="--", label='Validation')
            axe.xlabel('Epoch')
            axe.ylabel(name)
            # if metric == 'loss':
            #     plt.ylim([0, plt.ylim()[1]])
            # elif metric == 'auc':
            #     plt.ylim([0.8, 1])
            # else:
            #     plt.ylim([0, 1])

            plt.legend()
        plt.show()

    # def plot_confusion_matrix(self, results):
    #     conf_matrix = tf.math.confusion_matrix(labels=true_labels[:, 0], predictions=pred_labels[:, 0],
    #                                            num_classes=self.num_classes)
    #     _, ax = plt.subplots(figsize=(6, 4))
    #     sns.heatmap(conf_matrix.numpy() / (pred_labels.shape[0] / 2), annot=True, cmap=plt.cm.Blues)
    #     ax.set_ylabel('True label')
    #     ax.set_xlabel('Predicted label')
    #     ax.set_title("Confusion matrix", size=18)
    #     plt.show()

    @staticmethod
    def make_tensorboard():
        path = "logs/fit/"
        if os.path.exists(path):
            shutil.rmtree(path)
        log_dir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        return tensorboard

