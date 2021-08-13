import os
# Dataset parameters.
TRAIN_IMAGES_PATH = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/train'
TRAIN_LABELS_PATH= 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_train.csv'
VALIDATION_IMAGES_PATH = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
VALIDATION_LABELS_PATH = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'
TEST_IMAGES_PATH = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/test'
TEST_LABELS_PATH = 'D:/MIFI/SCIENTIFIC WORK/DATASETS/Russian road signs classification/rtsd-r1/gt_test.csv'


# Paths to data, saved logs and weights.
# DATA_PATH = 'data'
# LOGS_DIR = 'logs'
# WEIGHTS_PATH = 'weights'
# TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train_data.json')
# TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data.json')

# Training parameters.
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100

# Custom model parameters.
ALPHA = 1.0
REGULARIZATION = 0.0005
ACTIVATION_TYPE = 'leaky'

MODEL_TYPE = 'custom_resnet18'

NUM_CLASSES = 67
INPUT_SHAPE = (48, 48, 3)
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'

# augmentation configuration
AUG_CONFIG = {'flip': None, 'rotation': (-0.1, 0.1), 'crop': 0.9, 'translation': None,
              'zoom': 0.2, 'saturation': (0.5, 1.0), 'hue': 0.1, 'brightness': (0.3, 1.0), 'noise': 0.45, 'blur': 3.0,
              'contrast': 0.7, 'target_size': (48, 48)}
# metrics configuration
# METRIC_NAMES = ['accuracy', 'precision', 'recall', 'f1']
METRIC_NAMES = ['accuracy', 'precision', 'recall']
# service parameters
APPLY_SAMPLE_WEIGHT = True
SHOW_LEARNING_CURVES = True
