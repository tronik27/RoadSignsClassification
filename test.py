import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count
import time
import os
from typing import Tuple, Optional
import cv2
from tensorflow_addons.metrics import F1Score
from config import CLASS_NAMES


class DataGen(tf.keras.utils.Sequence):

    def __init__(self, data_path: str, batch_size: int, target_size: Tuple[int, int, int]) -> None:
        """
        Data generator for the task of road signs classifying.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param target_size: the size to which all images in the dataset are reduced.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.file_pathes = self.__get_images_path()

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Getting batch.
        :param index: batch number.
        :return: image tensor.
        """
        data_batch = self.file_pathes[index * self.batch_size:(index + 1) * self.batch_size]
        images = self.__get_data(data_batch)
        return images

    def __len__(self):
        return len(self.file_pathes) // self.batch_size

    def __get_images_path(self) -> list:
        """
        Getting pathes to images.
        :return: list containing the pathes to the files in the directory and subdirectories given by self.path.
        """
        file_pathes = []
        for path, subdirs, files in os.walk(self.data_path):
            for name in files:
                file_pathes.append(os.path.join(path, name))
        return file_pathes

    def __get_data(self, images_path: list) -> np.ndarray:
        """
        Making batch.
        :param images_path: list of pathes for images included in the batch.
        :return: image tensor.
        """
        images_batch = np.asarray([self.__get_image(path) for path in images_path])
        return images_batch

    def __get_image(self, path: str) -> np.ndarray:
        """
        Reads an image from a folder .
        :param path: path to the folder with images.
        :return: normalized image array.
        """
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size[:-1], interpolation=cv2.INTER_AREA)
        return image / 255


class RSCWork:

    def __init__(self, path_to_model: str, path_to_data: str, target_size: Tuple[int, int, int],) -> None:
        try:
            self.nn = tf.keras.models.load_model(path_to_model, custom_objects={"F1Score": F1Score})
        except FileNotFoundError:
            raise ValueError('There are no weights to save the trained model! Try to train the model first.')
        print(self.nn.summary())
        self.path_to_data = path_to_data
        self.target_size = target_size

    def predict(self, batch_size: int) -> list:
        """
        Validating model on test dataset, getting metrics and measuring inference time with batch size 1.
        Function will print validating results and measured time.
        """
        computation_time = 0.
        predicted_labels = []
        data_gen = DataGen(data_path=self.path_to_data, batch_size=batch_size, target_size=self.target_size)
        print('[INFO] predicting labels...')
        for i in range(len(data_gen)):
            img_batch = data_gen[i]
            start_time = time.time()
            predicts = self.nn.predict(img_batch, use_multiprocessing=True)
            finish_time = time.time()
            computation_time += finish_time - start_time
            predicted_labels += (np.argmax(predicts, axis=-1)).tolist()

        print('[INFO] Mean FPS: {:.04f}.'.format(len(predicted_labels) / computation_time))
        return predicted_labels

    def predict_and_show(self, class_names: list = []) -> None:

        data_gen = DataGen(data_path=self.path_to_data, batch_size=1, target_size=self.target_size)

        for i in range(len(data_gen)):
            img = data_gen[i]
            predict = self.nn(img, training=False)
            print(predict)
            prediction = np.max(predict)
            print(prediction)
            if class_names:
                label = class_names[int(np.argmax(predict))]
            else:
                label = int((np.argmax(predict)))
            plt.imshow(img[0, :, :, :])
            plt.title('Predicted Label: {}, \n confidence of prediction: {:.02f}%.'.format(label, prediction))
            plt.waitforbuttonpress(0)
            plt.close('all')


data = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/German road signs classification/Test'
model = 'D:/Resume/Classification/RoadSignsClassification/trained_model_1/small_image_net10'
classifier = RSCWork(path_to_model=model, path_to_data=data, target_size=(48, 48, 3))
# pred = classifier.predict(batch_size=1024)
# print(len(pred))
# print(pred)
classifier.predict_and_show(class_names=CLASS_NAMES)
