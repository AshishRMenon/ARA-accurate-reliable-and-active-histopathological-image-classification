# ARA-accurate-reliable-and-active-histopathological-image-classification
Accurate, reliable and active (ARA) image classification framework  is a new Bayesian Convolutional Neural Network (ARA-CNN) for classifying histopathological images of colorectal cancer

(https://www.nature.com/articles/s41598-019-50587-1)

The repo tries to implement the core idea of the paper (active learning method to fetch the labels for classification efficiently by employing variational dropouts) using pytorch framework with few changes in the implementation by making use of a squeezenet (squeezenet1_1) architechture for classification, instead of the custom model that is mentioned in the original paper, and provides promising results.

## Classification using baseline (squeezenet model):
python baseline_model.py &nbsp; --root_dir ${datasetfolder} &nbsp; --model_save_path ${model_save_path} &nbsp; --log_dir ${tensorboard_logging_dir}


## Generate PR and ROC curves (squeezenet model):
python generate_pr_roc_curves.py &nbsp; --root_dir ${datasetfolder} &nbsp; --model_load_path ${model_load_path} &nbsp; --log_dir ${tensorboard_logging_dir}



## Active learning experiment

python ara_active_learning.py &nbsp; --root_dir ${datasetfolder} &nbsp;-root_dir_copy ${datasetfolder_copy} &nbsp; --active_learning_dir ${active_learning_folder} &nbsp; --random_transfer_dir  ${random_expt_folder} &nbsp; --log_dir ${tensorboard_logging_dir}
                          
               
datasetfolder_copy : copy of the original datasetfolder without images (Should be prepared by copying folders of original dataset as is without any images)

active_learning_folder : directory to store dnamically changing files during active_learning (any random folder of your choice)

random_expt_folder  : directory to store dynamically changing files during random_expts (any random folder of your choice)

The script also has parameters to set the number of train epochs, test runs for active learning and variational dropout calls
