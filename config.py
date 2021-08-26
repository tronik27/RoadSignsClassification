# Paths to data.
PATH = 'E:/DATASETS/German road signs classification'
TRAIN_DF_NAME = 'Train.csv'
TEST_DF_NAME = 'Test.csv'
META_DF_NAME = 'Meta.csv'


# Paths to data, saved logs and weights.
# DATA_PATH = 'data'
# LOGS_DIR = 'logs'
# WEIGHTS_PATH = 'weights'
# TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train_data.json')
# TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data.json')

# Training parameters.
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 35

# Custom model parameters.
REGULARIZATION = 0.0005
NUM_FILTERS = 16

MODEL_NAME = 'small_image_net10'

NUM_CLASSES = 43
INPUT_SHAPE = (48, 48, 3)
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'

CLASS_NAMES = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
               'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
               'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
               'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry',
               'General caution', 'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road',
               'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians',
               'Children crossing', 'Bicycles crossing', 'Beware of ice/snow ', 'Wild animals crossing',
               'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
               'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
               'End of no passing', 'End no passing veh > 3.5 tons']

# augmentation configuration
AUG_CONFIG = ['crop', 'rotate', 'sharpen', 'rgb_shift', 'brightness_contrast', 'hue_saturation',
              'distortion', 'blur', 'noise']
# AUG_CONFIG = []
# metrics configuration
METRIC_NAMES = ['categorical_accuracy', 'f1_score']
# service parameters
APPLY_SAMPLE_WEIGHT = True
SHOW_LEARNING_CURVES = True
SHOW_DATASET_INFO = True
