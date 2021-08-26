# RoadSignsClassification
![img2.png](images/img2.PNG)

This repository contains the implementation of the road sign classifier. The German Traffic Sign Recognition Benchmark dataset (https://benchmark.ini.rub.de) was used for training and testing. It contains about 39000 training and 12500 test images belonging to 43 classes. The complexity of this dataset lies in the rather strong imbalance of the classes (see the figure below) so the class weights are applied during training.

![img3.png](images/img3.PNG)

GTSRB dataset has this configuration:
```
├───Meta
├───Test
└───Train
    ├───0
    ├───1
   ...
    ├───41
    └───42
```
When training a model, images for training are split into training and validation sets. Albumintations library (https://albumentations.ai/docs/api_reference/augmentations/) was used for image augmentation. As can be seen from the learning curves presented below, the classification quality metrics for the validation set were superior to those for the training set, which indicates that the model is not overfitting.

![img1.png](images/img1.PNG)

## Testing
This section presents the results of the trained model obtained on the test set: 

*Loss: 0.13592;*

*Categorical accuracy: 98.779%;*

*F1 score: 98.780%;*

Classification report:
```
                               precision    recall  f1-score   support

         Speed limit (20km/h)       0.97      1.00      0.98        60
         Speed limit (30km/h)       0.99      0.99      0.99       717
         Speed limit (50km/h)       0.99      1.00      0.99       750
         Speed limit (60km/h)       0.98      0.98      0.98       449
         Speed limit (70km/h)       1.00      0.99      0.99       660
         Speed limit (80km/h)       0.95      0.99      0.97       629
  End of speed limit (80km/h)       1.00      0.82      0.90       150
        Speed limit (100km/h)       1.00      0.99      0.99       450
        Speed limit (120km/h)       0.99      0.99      0.99       449
                   No passing       0.98      1.00      0.99       479
 No passing veh over 3.5 tons       1.00      1.00      1.00       659
 Right-of-way at intersection       0.98      1.00      0.99       418
                Priority road       1.00      0.99      0.99       688
                        Yield       1.00      1.00      1.00       719
                         Stop       1.00      1.00      1.00       270
                  No vehicles       0.96      0.99      0.97       210
    Veh > 3.5 tons prohibited       0.99      1.00      1.00       150
                     No entry       1.00      1.00      1.00       360
              General caution       1.00      0.95      0.97       389
         Dangerous curve left       1.00      1.00      1.00        60
        Dangerous curve right       0.95      1.00      0.97        90
                 Double curve       0.91      1.00      0.95        90
                   Bumpy road       1.00      1.00      1.00       120
                Slippery road       0.99      0.99      0.99       150
    Road narrows on the right       0.94      0.99      0.96        90
                    Road work       1.00      0.98      0.99       479
              Traffic signals       0.97      1.00      0.99       180
                  Pedestrians       0.92      1.00      0.96        60
            Children crossing       1.00      1.00      1.00       150
            Bicycles crossing       1.00      1.00      1.00        90
          Beware of ice/snow        0.98      0.89      0.93       150
        Wild animals crossing       0.97      0.99      0.98       270
   End speed + passing limits       1.00      1.00      1.00        60
             Turn right ahead       1.00      1.00      1.00       209
              Turn left ahead       0.99      1.00      1.00       119
                   Ahead only       1.00      1.00      1.00       389
         Go straight or right       1.00      1.00      1.00       119
          Go straight or left       0.98      1.00      0.99        60
                   Keep right       1.00      1.00      1.00       688
                    Keep left       0.98      0.98      0.98        90
         Roundabout mandatory       1.00      0.94      0.97        90
            End of no passing       0.81      0.85      0.83        60
End no passing veh > 3.5 tons       1.00      0.87      0.93        89

                     accuracy                           0.99     12608
                    macro avg       0.98      0.98      0.98     12608
                 weighted avg       0.99      0.99      0.99     12608
```
Examples of image classification by model:

![img4.png](images/img4.PNG)

## Using script

If you want to train/evaluate or save the model by yourself, then use the code presented in **main.py**. The main parameters used in the code are specified in the file **config.py**. For image classification by pretrained model use **test.py**. The trained model is stored in the folder **trained_model/small_image_net10**. Below are examples of image classification by the trained model:
