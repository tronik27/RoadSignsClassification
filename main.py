from train_and_test import RoadSignsClassification
from config import BATCH_SIZE, INPUT_SHAPE, METRIC_NAMES, NUM_CLASSES, TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH,\
    VALIDATION_IMAGES_PATH, VALIDATION_LABELS_PATH, TEST_IMAGES_PATH, TEST_LABELS_PATH, NUM_EPOCHS, AUG_CONFIG,\
    APPLY_SAMPLE_WEIGHT, SHOW_LEARNING_CURVES

classifier = RoadSignsClassification(
    batch_size=BATCH_SIZE,
    target_size=INPUT_SHAPE,
    metric_names=METRIC_NAMES,
    num_classes=NUM_CLASSES
)
classifier.train(
    images=TRAIN_IMAGES_PATH,
    labels=TRAIN_LABELS_PATH,
    validation_images=VALIDATION_IMAGES_PATH,
    validation_labels=VALIDATION_LABELS_PATH,
    epochs=NUM_EPOCHS,
    apply_sample_weight=APPLY_SAMPLE_WEIGHT,
    augmentation=AUG_CONFIG,
    show_learning_curves=SHOW_LEARNING_CURVES
)
