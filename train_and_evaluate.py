import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow_addons.metrics import F1Score
import matplotlib.pyplot as plt
from DataPreprocessing import CustomDataGen
from cnn_model import SmallImageNet
import pandas as pd
import numpy as np
from multiprocessing import cpu_count
import random
import os
from typing import Tuple, Optional
from sklearn.metrics import classification_report


class RoadSignsClassification:
    def __init__(self,
                 batch_size: int,
                 target_size: Tuple[int, int, int],
                 metric_names: list,
                 num_classes: int,
                 regularization: Optional[float],
                 model_name: str,
                 class_names: list,
                 input_name: str,
                 output_name: str,
                 num_filters: int,
                 learning_rate: float,
                 path_to_model_weights: str = 'road_signs_model/weights') -> None:
        """
        Road signs classifier class.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param target_size: the size to which all images in the dataset are reduced.
        :param metric_names: list of metric which will be calculated during training and evaluation.
        :param num_classes: number of classes of images in dataset.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param model_name: name of model.
        :param class_names: class names.
        :param input_name: name of the input tensor.
        :param output_name: name of the output tensor.
        :param num_filters: network expansion factor, determines the number of filters in start layer.
        :param learning_rate: learning rate when training the model.
        :param path_to_model_weights: folder where the weights of the model will be saved after the epoch at which it
         showed the best result.
        """
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
              show_dataset_info: bool = False
              ) -> None:
        """
        Method for training the model.
        :param path: path to folder containing data.
        :param file_name: name of file containing dataframe.
        :param valid_size: part of the data that will be allocated to the validation set.
        :param augmentation: list of transforms to be applied to the training image.
        :param epochs: number of epochs to train the model.
        :param apply_sample_weight: indicates whether to apply sample weights or not.
        :param show_learning_curves: indicates whether to show show learning curves or not.
        :param show_image_data: indicates whether to show original and augmented image with labels or not.
        :param show_dataset_info: indicates whether to show information about dataset or not.
        """
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
            callbacks=self.__get_callbacks(),
            epochs=epochs,
            workers=cpu_count(),
        )

        if show_learning_curves:
            print('[INFO] displaying information about learning process. Close the window to continue...')
            self.__plot_learning_curves(history)

    def evaluate(self, path: str, file_name: str, show_image_data: bool) -> None:
        """
        Method for evaluating a model on a test set.
        :param path: path to dataframe.
        :param file_name: name of file containing dataframe.
        :param show_image_data: indicates whether to show predictions for set images or not.
        """
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
            print('{}: {:.03f}'.format(metric, results[i]))
        self.__evaluation_report(test_datagen)
        if show_image_data:
            self.__show_image_data(test_datagen)

    def save_model(self, path_to_save: str) -> None:
        """
        Method for saving yhe whole model.
        :param path_to_save: folder where the model will be stored.
        """
        print('[INFO] saving network model...')
        try:
            self.nn.load_weights(self.path_to_model_weights)
        except FileNotFoundError:
            raise ValueError('There are no weights to save the trained model! Try to train the model first.')

        self.nn.save(os.path.join(path_to_save, self.model_name))

    def __show_image_data(self, generator: tf.keras.utils.Sequence) -> None:
        """
        Method for showing predictions.
        :param generator: data generator.
        """
        j = random.randint(0, len(generator))
        images, labels = generator[j]
        labels = np.argmax(labels, axis=-1)
        if images.shape[0] > 5:
            images = images[:5, :, :, :]
            labels = labels[:5]
        predict_labels = np.argmax(self.nn.predict(images), axis=-1)
        predictions = np.max(self.nn.predict(images), axis=-1)

        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
        fig.suptitle('Network prediction results:', fontsize=14, fontweight="bold")
        for i in range(images.shape[0]):
            text = 'True label:\n {},\n Predicted label:\n {},\n Confidence of prediction:\n {:.02f}%.'.format(
                self.class_names[labels[i]],
                self.class_names[predict_labels[i]],
                predictions[i] * 100
            )
            axes[i].imshow(images[i, :, :, :])
            axes[i].set_title(text, size=9)
            axes[i].axis('off')
        plt.show()

    def __get_callbacks(self) -> list:
        """
        Method for creating a list of callbacks.
        :return: list containing callbacks.
        """

        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callbacks = list()
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.01,
                                                         factor=0.5, patience=3, min_lr=0.00001)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.path_to_model_weights, save_weights_only=True,
                                                        save_best_only=True, monitor='val_loss', mode='min')
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)
        callbacks += [reduce_lr, checkpoint, stop]
        return callbacks

    def __get_metrics(self) -> list:
        """
        Method for creating a list of metrics according to the configuration.
        :return: list containing metric objects.
        """
        metrics = {'categorical_accuracy': CategoricalAccuracy(),
                   'precision': Precision(),
                   'recall': Recall(),
                   'f1_score': F1Score(num_classes=self.num_classes, threshold=0.5, average='micro')}
        metric_list = list()
        for mertic in self.metric_names:
            metric_list.append(metrics[mertic])
        return metric_list

    def __plot_learning_curves(self, metric_data) -> None:
        """
        Method for plotting learning curves.
        :param metric_data: dictionary containing metric an loss logs.
        """
        print(metric_data)
        figure, axes = plt.subplots(len(metric_data.history) // 2, 1, figsize=(5, 10))
        for axe, metric in zip(axes, self.nn.metrics_names):
            name = metric.replace("_", " ").capitalize()
            axe.plot(metric_data.epoch, metric_data.history[metric], label='Train')
            axe.plot(metric_data.epoch, metric_data.history['val_' + metric], linestyle="--",
                     label='Validation')
            axe.set_xlabel('Epoch')
            axe.set_ylabel(name)
            axe.grid(color='coral', linestyle='--', linewidth=0.5)
            axe.legend()
        plt.show()

    def __evaluation_report(self, generator: tf.keras.utils.Sequence) -> None:
        """
        Method for creating a pivot table showing the quality of the classification for different classes.
        :param generator: data generator.
        """
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
        """
        Method of splitting a data frame into data frames for training and validation.
        :param path: path to dataframe.
        :param file_name: name of file containing dataframe.
        :param valid_size: part of the dataframe that will be allocated to the validation set.
        :return: train and validation data frames.
        """
        df = pd.read_csv(os.path.join(path, file_name))
        df = df.sample(frac=1).reset_index(drop=True)
        train_df = df.iloc[:-int(valid_size * len(df)), :]
        valid_df = df.iloc[-int(valid_size * len(df)):, :]
        return train_df, valid_df

