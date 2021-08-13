import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Tuple
from Augmentation import image_augmentation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, meta_path: str, path_to_image: str, batch_size: int, target_size: Tuple[int, int, int],
                 train_mode: bool = True, aug_config: dict = {}, apply_weights: bool = True) -> None:
        """
        Data generator for the task of road signs classifying.
        :param meta_path: path to dataframe with metadata.
        :param path_to_image: path to the folder with images.
        :param batch_size: size of the batch of images fed to the input of the neural network.
        :param target_size: the size to which all images in the dataset are reduced.
        :param train_mode: indicates whether the generator is being used to create a training or test dataset.
        :param aug_config: a dictionary containing the parameter values for augmentation.
        :param apply_weights: indicates whether to apply weights to balance classes or not.
        """
        self.df = pd.read_csv(meta_path)
        self.batch_size = batch_size
        self.target_size = target_size
        self.path_to_image = path_to_image
        if train_mode:
            self.augmentation = image_augmentation(config=aug_config)
        else:
            self.augmentation = None
        self.number_of_images = len(self.df)
        self.num_classes = self.df['class_number'].nunique()
        if apply_weights:
            self.class_weights = compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(self.df['class_number']),
                                                      y=self.df['class_number'])
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
        if self.augmentation:
            images = self.augmentation(images)
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
        images_path = batch_df['filename']
        labels = batch_df['class_number']

        images_batch = np.asarray([self.__get_image(self.path_to_image + '/' + path) for path in images_path])
        labels_batch = self.__get_labels(labels)
        return images_batch, labels_batch

    def __get_image(self, path: np.ndarray) -> np.ndarray:
        """
        Reads an image from a folder .
        :param path: path to the folder with images.
        :return: normalized image array.
        """
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (self.target_size[0], self.target_size[1])).numpy()
        return image_arr / 255.

    def __get_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Converts a class labels to binary class matrix.
        :param labels: class labels of images included in the batch.
        :return: binary matrix representation of labels.
        """
        return tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

    def __get_sample_weights(self, labels: np.ndarray) -> np.ndarray:
        """
        Gets weights for objects of different classes .
        :param labels: class labels of images included in the batch.
        :return: array of weights according to class labels.
        """
        labels = np.argmax(labels, axis=-1)
        return self.class_weights[labels]

    def show_image_data(self, num_of_images: int = 8) -> None:
        """
        Method for showing original and augmented image with labels.
        :param num_of_images: number of images to display. The maximum number of images available for display  - 8.
        """
        if num_of_images > 8:
            num_of_images = 8
        images, labels = self.__get_data(self.df.sample(n=num_of_images))
        aug_images = self.augmentation(images)
        labels = np.argmax(labels, axis=-1)
        fig, axes = plt.subplots(nrows=2, ncols=num_of_images, figsize=(2 * num_of_images, 4))
        for i in range(num_of_images):
            axes[0][i].imshow(images[i, :, :, :])
            axes[0][i].set_title('Original, class: "{}"'.format(labels[i]), size=9)
            axes[0][i].axis('off')
            axes[1][i].imshow(aug_images[i, :, :, :])
            axes[1][i].set_title('Augmented, class: "{}"'.format(labels[i]), size=9)
            axes[1][i].axis('off')
        plt.show()

    def dataset_info(self) -> None:
        """
        Method for plotting the distribution of the number of images by class.
        """
        sns_plot = sns.histplot(self.df['class_number'], bins=self.num_classes, color='blueviolet')
        sns_plot.set_title('Number of images per class (there are {} classes in dataset)'.format(self.num_classes))
        sns_plot.set_xlabel('Class')
        sns_plot.set_ylabel('Number of images')
        sns_plot.grid(color='coral', linestyle='--', linewidth=0.5)
        sns_plot.set_xticks(np.arange(0, self.num_classes, 5))
        plt.show()


# metadata_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
# image_path = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
# AUG_CONF = {'flip': None, 'rotation': (-0.1, 0.1), 'crop': 0.9, 'translation': None,
#             'zoom': 0.2, 'saturation': (0.5, 1.0), 'hue': 0.1, 'brightness': (0.3, 1.0), 'noise': 0.45, 'blur': 3.0,
#             'contrast': 0.7, 'target_size': (48, 48)}
#
# datagen = CustomDataGen(meta_path=metadata_path, path_to_image=image_path, batch_size=32, target_size=(48, 48, 3),
#                         train_mode=True, aug_config=AUG_CONF, apply_weights=True)
# datagen.show_image_data()
# datagen.dataset_info()
# x, y, w = datagen[0]
# print(x.shape, y.shape, w)
