# RoadSignsClassification
![img2.png](images/img2.PNG)

This repository contains the implementation of the road sign classifier. The German Traffic Sign Recognition Benchmark dataset (https://benchmark.ini.rub.de) was used for training and testing. It contains about 39000 training and 12500 test images belonging to 43 classes. The complexity of this dataset lies in the rather strong imbalance of the classes (see the figure below) so the class weights are applied during training.

![img3.png](images/img3.PNG)

'''
├───Meta
├───Test
└───Train
    ├───0
    ├───1
   ...
    ├───41
    └───42
'''
![img1.png](images/img1.PNG)

![img4.png](images/img4.PNG)
