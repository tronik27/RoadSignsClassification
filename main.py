from train_and_evaluate import RoadSignsClassification
from config import BATCH_SIZE, INPUT_SHAPE, METRIC_NAMES, NUM_CLASSES, PATH, NUM_FILTERS, LEARNING_RATE, MODEL_NAME,\
    TRAIN_DF_NAME, TEST_DF_NAME, REGULARIZATION, NUM_EPOCHS, AUG_CONFIG, APPLY_SAMPLE_WEIGHT, META_DF_NAME, MODEL_PATH,\
    SHOW_LEARNING_CURVES, CLASS_NAMES, SHOW_DATASET_INFO, SHOW_IMAGE_DATA, INPUT_NAME, OUTPUT_NAME, WEIGHTS_PATH


def main():
    #  Creating the road signs classifier
    classifier = RoadSignsClassification(
        batch_size=BATCH_SIZE,
        target_size=INPUT_SHAPE,
        metric_names=METRIC_NAMES,
        num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
        model_name=MODEL_NAME,
        class_names=CLASS_NAMES,
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        path_to_model_weights=WEIGHTS_PATH
    )
    # Training classifier
    classifier.train(
        path=PATH,
        file_name=TRAIN_DF_NAME,
        epochs=NUM_EPOCHS,
        apply_sample_weight=APPLY_SAMPLE_WEIGHT,
        augmentation=AUG_CONFIG,
        show_learning_curves=SHOW_LEARNING_CURVES,
        show_image_data=META_DF_NAME,
        show_dataset_info=SHOW_DATASET_INFO
    )
    #  Saving trained classifier model
    classifier.save_model(path_to_save=MODEL_PATH)
    #  Testing classifier
    classifier.evaluate(
        path=PATH,
        file_name=TEST_DF_NAME,
        show_image_data=SHOW_IMAGE_DATA
    )


if __name__ == '__main__':
    main()
