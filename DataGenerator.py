import tensorflow as tf
import numpy as np
import pandas as pd


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, df, path_to_image, batch_size, target_size=(48, 48, 3), shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.path_to_image = path_to_image
        self.number_of_images = len(self.df)
        self.num_classes = df['class_number'].nunique()

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        data_batch = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = self.__get_data(data_batch)
        return images, labels

    def __len__(self):
        return self.number_of_images // self.batch_size

    def __get_data(self, batch_df):
        images_path = batch_df['filename']
        labels = batch_df['class_number']

        images_batch = np.asarray([self.__get_image(self.path_to_image + path) for path in images_path])
        labels_batch = self.__get_labels(labels)

        return images_batch, labels_batch

    def __get_image(self, path):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (self.target_size[0], self.target_size[1])).numpy()
        return image_arr / 255.

    def __get_labels(self, labels):
        return tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)


df = pd.read_csv('D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv')
datagen = CustomDataGen(df,
                        path_to_image='D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train',
                        batch_size=32, target_size=(48, 48, 3))
x, y = datagen[0]
print(x.shape, y.shape)
