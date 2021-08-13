import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from random import choice
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast, RandomFlip, RandomRotation, RandomCrop,\
    RandomTranslation, RandomZoom, Resizing


class RandomChangeSaturation(Layer):
    def __init__(self, lower=0.5, upper=1.0, **kwargs) -> None:
        """
        Layer for the data augmentation by random hue change.
        :param lower: The maximum value for brightness change. Must be in the interval [0, 0.5]
        :param upper: The maximum value for brightness change. Must be in the interval [0, 0.5]
        """
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, tensor) -> tf.Tensor:
        """
        Performs the logic of applying the layer to the input tensors.
        :param tensor: input tensor of images
        """
        tensor = tf.map_fn(lambda x: tf.image.random_saturation(image=x, lower=self.lower, upper=self.upper), tensor)
        return tensor


class RandomChangeHue(Layer):
    def __init__(self, factor: float = 0.2, **kwargs) -> None:
        """
        Layer for the data augmentation by random hue change.
        :param factor: The maximum value for brightness change. Must be in the interval [0, 0.5]
        """
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, tensor) -> tf.Tensor:
        """
        Performs the logic of applying the layer to the input tensors.
        :param tensor: input tensor of images
        """
        tensor = tf.map_fn(lambda x: tf.image.random_hue(image=x, max_delta=self.factor), tensor)
        return tensor


class RandomChangeBrightness(Layer):
    def __init__(self, factor: float = 0.2, **kwargs) -> None:
        """
        Layer for the data augmentation by random brightness change.
        :param factor: The maximum value for brightness change. Must be non-negative.
        """
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, tensor) -> tf.Tensor:
        """
        Performs the logic of applying the layer to the input tensors.
        :param tensor: input tensor of images
        """
        tensor = tf.map_fn(lambda x: tf.keras.preprocessing.image.random_brightness(x=x, brightness_range=self.factor),
                           tensor)
        return tensor


class AddingNoise(Layer):
    def __init__(self, factor: float = 0.2, **kwargs) -> None:
        """
        Layer for the data augmentation by adding noise.
        :param factor: standard deviation of noise. Must be non-negative.
        """
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, tensor) -> tf.Tensor:
        """
        Performs the logic of applying the layer to the input tensors.
        :param tensor: input tensor of images
        """
        return self.add_noise(tensor)

    def add_noise(self, images: tf.Tensor) -> tf.Tensor:
        """
        Adds randomly selected noise to images.
        :param images: tensor of images
        """
        gauss_noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=self.factor, dtype=tf.float32)
        uniform_noise = tf.random.uniform(shape=tf.shape(images), minval=0, maxval=self.factor, dtype=tf.dtypes.float32)
        noise = choice([gauss_noise, uniform_noise])
        noisy_images = tf.add(images, noise)
        return noisy_images / tf.reduce_max(noisy_images, keepdims=True)


class AddingGaussianBlur(Layer):
    def __init__(self, factor: float = 3.0, **kwargs) -> None:
        """
        Layer for the data augmentation by adding blur.
        :param factor: standard deviation of gaussian distribution. Must be non-negative.
        """
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, tensor) -> tf.Tensor:
        """
        Performs the logic of applying the layer to the input tensors.
        :param tensor: input tensor of images
        """
        tensor = tf.map_fn(lambda x: self.add_blur(image=x), tensor)
        return tensor

    def add_blur(self, image: tf.Tensor) -> tf.Tensor:
        """
        Adds gaussian blur to images.
        :param image: image tensor.
        """
        def gaussian_kernel(size, std):
            x_range = tf.range(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
            y_range = tf.range((size - 1) // 2, -(size - 1) // 2 - 1, -1)

            xs, ys = tf.meshgrid(x_range, y_range)
            xs = tf.cast(xs, tf.float32)
            ys = tf.cast(xs, tf.float32)
            k = tf.exp(-(xs ** 2 + ys ** 2) / (2 * (std ** 2))) / (2 * np.pi * (std ** 2))
            return tf.cast(k / tf.reduce_sum(k), tf.float32)

        kernel = gaussian_kernel(size=3, std=tf.random.uniform(shape=[], maxval=self.factor, dtype=tf.float32))
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

        r, g, b = tf.split(image, [1, 1, 1], axis=-1)
        r_blur = tf.nn.conv2d(tf.expand_dims(r, axis=0), kernel, [1, 1, 1, 1], 'SAME')
        g_blur = tf.nn.conv2d(tf.expand_dims(g, axis=0), kernel, [1, 1, 1, 1], 'SAME')
        b_blur = tf.nn.conv2d(tf.expand_dims(b, axis=0), kernel, [1, 1, 1, 1], 'SAME')

        blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
        return tf.squeeze(blur_image, axis=0)


def image_augmentation(config):
    """
    Adds gaussian blur to images.
    :param config: dictionary with a set of parameters for augmentation.
    """
    augmentation = tf.keras.Sequential()
    if config['crop']:
        augmentation.add(RandomCrop(height=int(config['crop'] * config['target_size'][0]),
                                    width=int(config['crop'] * config['target_size'][1])))
        augmentation.add(Resizing(height=config['target_size'][0], width=config['target_size'][1]))
    if config['zoom']:
        augmentation.add(RandomZoom(config['zoom']))
    if config['translation']:
        augmentation.add(RandomTranslation(height_factor=config['translation'],
                                           width_factor=config['translation'],
                                           fill_mode='wrap'))
    if config['flip']:
        augmentation.add(RandomFlip(config['flip']))
    if config['rotation']:
        augmentation.add(RandomRotation(config['rotation']))
    if config['contrast']:
        augmentation.add(RandomContrast(factor=config['contrast']))
    if config['saturation']:
        augmentation.add(RandomChangeSaturation(upper=config['saturation'][1], lower=config['saturation'][0]))
    if config['hue']:
        augmentation.add(RandomChangeHue(config['hue']))
    if config['brightness']:
        augmentation.add(RandomChangeBrightness(config['brightness']))
    if config['blur']:
        augmentation.add(AddingGaussianBlur(config['blur']))
    if config['noise']:
        augmentation.add(AddingNoise(config['noise']))
    return augmentation
