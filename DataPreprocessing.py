import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from Augmentation import image_augmentation
from sklearn.utils.class_weight import compute_class_weight
import cv2
import albumentations as aug
import os


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, data_path: str, data: pd.DataFrame, batch_size: int, target_size: Tuple[int, int, int],
                 aug_config: list = [], apply_weights: bool = False) -> None:
        """
        Data generator for the task of road signs classifying.
        :param data:  dataframe with metadata.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param target_size: the size to which all images in the dataset are reduced.
        :param aug_config: a dictionary containing the parameter values for augmentation.
        :param apply_weights: indicates whether to apply weights to balance classes or not.
        """
        self.data_path = data_path
        self.df = data
        self.batch_size = batch_size
        self.target_size = target_size
        self.images_column = 'Path'
        self.labels_column = 'ClassId'
        self.aug_config = aug_config
        self.number_of_images = len(self.df)
        self.num_classes = self.df[self.labels_column].nunique()
        if apply_weights:
            self.class_weights = compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(self.df[self.labels_column]),
                                                      y=self.df[self.labels_column])
        else:
            self.class_weights = np.array([])

    def on_epoch_end(self):
        """
        Random shuffling of training data at the end of each epoch during training.
        """
        if self.augmentation:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Getting batch.
        :param index: batch number.
        :return: image and labels or image, labels and sample weights tensors.
        """
        data_batch = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = self.__get_data(data_batch)
        if self.class_weights.any():
            sample_weights = self.__get_sample_weights(labels)
            return images, labels, sample_weights
        else:
            return images, labels

    def __len__(self):
        return self.number_of_images // self.batch_size

    def __get_data(self, batch_df) -> Tuple[np.ndarray, np.ndarray]:
        """
        Making batch.
        :param batch_df: part of the dataframe containing metadata for images included in the batch.
        :return: image and labels tensors.
        """
        images_path = batch_df[self.images_column]
        labels = batch_df[self.labels_column].to_numpy()
        images_batch = np.asarray([self.__get_image(self.data_path + '/' + path) for path in images_path])
        labels_batch = self.__get_labels(labels)
        return images_batch, labels_batch

    def __get_image(self, path: str) -> np.ndarray:
        """
        Reads an image from a folder .
        :param path: path to the folder with images.
        :return: normalized image array.
        """
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size[:-1], interpolation=cv2.INTER_AREA)
        if self.aug_config and self.aug_config:
            image = self.augmentation(image)
        preprocessed_image = aug.Compose(
            [aug.CLAHE(tile_grid_size=(10, 10), always_apply=True, clip_limit=1.0)]
        )(image=image)['image']
        return preprocessed_image / 255

    def __get_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Converts a class labels to binary class matrix.
        :param labels: class labels of images included in the batch.
        :return: binary matrix representation of labels.
        """
        return tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

    def augmentation(self, image: np.array) -> np.array:
        """
        Apply augmentation to the image.
        :param image: image array.
        :return: augmented image.
        """
        augmentation = image_augmentation(config=self.aug_config, target_shape=self.target_size)
        return augmentation(image=image)['image']

    def __get_sample_weights(self, labels: np.ndarray) -> np.ndarray:
        """
        Gets weights for objects of different classes .
        :param labels: class labels of images included in the batch.
        :return: array of weights according to class labels.
        """
        labels = np.argmax(labels, axis=-1)
        return self.class_weights[labels]

    def show_image_data(self, meta_file_name: str, class_names: list, num_of_images: int = 8) -> None:
        """
        Method for showing original and augmented image with labels.
        :param num_of_images: number of images to display. The maximum number of images available for display  - 8.
        :param meta_file_name: path to info data.
        :param class_names: class names.
        """
        def get_images(df):
            image = []
            for path in df:
                img = cv2.cvtColor(cv2.imread(self.data_path + '/' + path), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.target_size[:-1], interpolation=cv2.INTER_AREA)
                image.append(img)
            image = np.asarray(image)
            return image

        if num_of_images > 8:
            num_of_images = 8

        batch_df = self.df.sample(n=num_of_images)
        meta_df = pd.read_csv(os.path.join(self.data_path, meta_file_name))

        images = get_images(batch_df[self.images_column])
        augmented_images, labels = self.__get_data(batch_df)
        labels = np.argmax(labels, axis=-1)

        fig, axes = plt.subplots(nrows=3, ncols=num_of_images, figsize=(2 * num_of_images, 6))
        fig.suptitle('Reference,original and augmented images', fontsize=16)
        for i in range(num_of_images):
            meta_path = meta_df.loc[meta_df[self.labels_column] == labels[i]][self.images_column].to_list()
            axes[0][i].imshow(cv2.resize(cv2.cvtColor(cv2.imread(self.data_path + '/' + meta_path[0]),
                                                      cv2.COLOR_BGR2RGB), (100, 100), interpolation=cv2.INTER_AREA))
            axes[0][i].set_title('{}'.format(class_names[labels[i]]), size=9)
            axes[0][i].axis('off')
            axes[1][i].imshow(images[i, :, :, :])
            axes[1][i].set_title('Original, class: "{}"'.format(labels[i]), size=9)
            axes[1][i].axis('off')
            axes[2][i].imshow(augmented_images[i, :, :, :])
            axes[2][i].set_title('Augmented', size=9)
            axes[2][i].axis('off')
        plt.show()

    def show_dataset_info(self) -> None:
        """
        Method for plotting the distribution of the number of images by class.
        """
        sns_plot = sns.histplot(self.df[self.labels_column], bins=self.num_classes, color='blueviolet')
        sns_plot.set_title('Number of images per class \n (there are {} images belonging to {} classes'
                           ' in dataset)'.format(self.number_of_images, self.num_classes))
        sns_plot.set_xlabel('Class')
        sns_plot.set_ylabel('Number of images')
        sns_plot.grid(color='coral', linestyle='--', linewidth=0.5, axis='y')
        sns_plot.set_xticks(np.arange(0, self.num_classes, 3))
        plt.show()


# data = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/German road signs classification/Train.csv'
# metadata_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/German road signs classification/Meta.csv'
# AUG_CONF = ['crop', 'rotate', 'sharpen', 'rgb_shift', 'brightness_contrast', 'hue_saturation',
#             'blur', 'noise']
# datagen = CustomDataGen(data_path=data, batch_size=32, target_size=(48, 48, 3),
#                         aug_config=AUG_CONF, apply_weights=True)
# datagen.show_image_data(num_of_images=8, meta_path=metadata_path)
# # datagen.dataset_info()
# x, y, w = datagen[100]
# print(x.shape, y.shape, w)
