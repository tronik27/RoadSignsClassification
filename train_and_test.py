import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow_addons.metrics import F1Score
import matplotlib.pyplot as plt
from DataPreprocessing import CustomDataGen
from cnn_model import SmallImageNet
import pandas as pd
import numpy as np
from multiprocessing import cpu_count
import keras2onnx
import os
from typing import Tuple, Optional
from sklearn.metrics import classification_report


class RoadSignsClassification:
    def __init__(self, batch_size: int, target_size: Tuple[int, int, int], metric_names: list, num_classes: int,
                 regularization: Optional[float], model_name: str, class_names: list, input_name: str = 'input',
                 output_name: str = 'output', num_filters: int = 32, learning_rate: float = 0.001,
                 path_to_model_weights: str = 'road_signs_model/weights') -> None:
        self.metric_names = metric_names
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.path_to_model_weights = path_to_model_weights
        self.nn = SmallImageNet(
            input_shape=self.target_size, num_classes=num_classes, num_filters=num_filters,
            regularization=regularization, input_name=input_name, output_name=output_name
        ).build()
        self.nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=self.__get_metrics())
        self.model_name = model_name
        self.model_summary = self.nn.summary()

    def train(self,
              path: str,
              file_name: str,
              valid_size: float = 0.1,
              augmentation: list = [],
              epochs: int = 100,
              apply_sample_weight: bool = True,
              show_learning_curves: bool = False,
              show_image_data: str = '',
              show_dataset_info: bool = False,
              min_delta: float = 0.0001,
              scheduler=None) -> None:

        train_df, valid_df = self.__train_validation_split(path=path, file_name=file_name, valid_size=valid_size)
        train_datagen = CustomDataGen(data=train_df,
                                      data_path=path,
                                      batch_size=self.batch_size,
                                      target_size=self.target_size,
                                      aug_config=augmentation,
                                      apply_weights=apply_sample_weight)
        validation_datagen = CustomDataGen(data=valid_df,
                                           data_path=path,
                                           batch_size=self.batch_size,
                                           target_size=self.target_size,
                                           apply_weights=apply_sample_weight)

        if show_image_data:
            print('[INFO] displaying images from dataset. Close the window to continue...')
            train_datagen.show_image_data(meta_file_name=show_image_data, class_names=self.class_names)
        if show_dataset_info:
            print('[INFO] displaying information about dataset. Close the window to continue...')
            train_datagen.show_dataset_info()

        print('[INFO] training network...')
        history = self.nn.fit(
            train_datagen,
            validation_data=validation_datagen,
            steps_per_epoch=train_datagen.number_of_images // self.batch_size,
            callbacks=self.__get_callbacks(min_delta=min_delta, scheduler=scheduler),
            epochs=epochs,
            workers=cpu_count(),
        )

        if show_learning_curves:
            print('[INFO] displaying information about learning process. Close the window to continue...')
            self.__plot_learning_curves(history)

    def evaluate(self, path: str, file_name: str):

        test_df = pd.read_csv(os.path.join(path, file_name))

        test_datagen = CustomDataGen(
            data=test_df,
            data_path=path,
            batch_size=self.batch_size,
            target_size=self.target_size,
            apply_weights=False
        )

        try:
            self.nn.load_weights(self.path_to_model_weights)
        except FileNotFoundError:
            raise ValueError('There are no weights to evaluate the trained model! Try to train the model first.')

        print('[INFO] evaluating network...')
        results = self.nn.evaluate(test_datagen, batch_size=self.batch_size, verbose=0, use_multiprocessing=True)

        for i, metric in enumerate(self.nn.metrics_names):
            print('{}: {}'.format(metric, results[i]))
        self.__evaluation_report(test_datagen)

    def save_model(self, path_to_save: str) -> None:
        print('[INFO] saving network model...')
        try:
            self.nn.load_weights(self.path_to_model_weights)
        except FileNotFoundError:
            raise ValueError('There are no weights to save the trained model! Try to train the model first.')

        self.nn.save(os.path.join(path_to_save, self.model_name), save_format='h5')

    def __get_callbacks(self, min_delta, scheduler=None):
        callbacks = []
        if scheduler:
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
            callbacks.append(lr_scheduler)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.01,
                                                         factor=0.5, patience=1, min_lr=0.00001)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.path_to_model_weights, save_weights_only=True,
                                                        save_best_only=True, monitor='val_loss', mode='min')
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=min_delta)
        callbacks += [reduce_lr, checkpoint, stop]
        return callbacks

    def __get_metrics(self):
        metrics = {'categorical_accuracy': CategoricalAccuracy(),
                   'precision': Precision(),
                   'recall': Recall(),
                   'f1_score': F1Score(num_classes=self.num_classes, threshold=0.5, average='micro')}
        metric_list = list()
        for mertic in self.metric_names:
            metric_list.append(metrics[mertic])
        return metric_list

    def __plot_learning_curves(self, metric_data) -> None:
        figure, axes = plt.subplots(len(metric_data.history) // 2, 1, figsize=(5, 10))
        for axe, metric in zip(axes, self.nn.metrics_names):
            name = metric.replace("_", " ").capitalize()
            axe.plot(metric_data.epoch, metric_data.history[metric], label='Train')
            axe.plot(metric_data.epoch, metric_data.history['val_' + metric], linestyle="--", label='Validation')
            axe.set_xlabel('Epoch')
            axe.set_ylabel(name)
            axe.grid(color='coral', linestyle='--', linewidth=0.5)
            axe.legend()
        plt.show()

    def __evaluation_report(self, generator) -> None:
        print('[INFO] making classification report...')
        labels = np.array([])
        predictions = np.array([])
        for i in range(generator.number_of_images // self.batch_size):
            images, label = generator[i]
            predictions = np.hstack((predictions, np.argmax(self.nn.predict(images), axis=-1)))
            labels = np.hstack((labels, np.argmax(label, axis=-1)))
        print('[INFO] classification report:')
        print(classification_report(labels, predictions, target_names=self.class_names))

    @staticmethod
    def __train_validation_split(path: str, file_name: str, valid_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(os.path.join(path, file_name))
        df = df.sample(frac=1).reset_index(drop=True)
        train_df = df.iloc[:-int(valid_size * len(df)), :]
        valid_df = df.iloc[-int(valid_size * len(df)):, :]
        return train_df, valid_df

