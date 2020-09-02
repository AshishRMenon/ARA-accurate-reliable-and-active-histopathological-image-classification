# ARA-accurate-reliable-and-active-histopathological-image-classification
Accurate, reliable and active (ARA) image classification framework  is a new Bayesian Convolutional Neural Network (ARA-CNN) for classifying histopathological images of colorectal cancer

(https://www.nature.com/articles/s41598-019-50587-1)

The repo tries to implement the core idea of the paper (active learning method to fetch the labels for classification efficiently by employing variational dropouts) using pytorch framework with few changes in the implementation by making use of a squeezenet (squeezenet1_1) architechture for classification, instead of the custom model that is mentioned in the original paper, and provides promising results.
