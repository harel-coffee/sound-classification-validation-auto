# sound-classification-validation

Feature selection, training and validation for the ADORE sound classifier

## Prerequisites
To run the code you need:

 - libXtract Python bindings, source available https://github.com/jamiebullock/LibXtract. Make sure to install Swig beforehand or Python bindings will not be built.
 - sox for sound preprocessing

## Workflow

 - Download the dataset
 - Process the dataset using `prepare_audio.sh`
 - Calculate features in a given dataset using `feature_calculation.py`. Choose saving location inside the script.
 - Train the classifier with feature selection using `feature_selection.py`.
 - Once you have the classifier and feature list, you can do different validation steps

### Validation

 - Perform cross validation on the dataset using `generate_conf_matrices_train_test.py`.
 - Record new dataset using a robot with `play_record.py`, then calculate features on that dataset (using `feature_calculation.py`) and validate the existing classifier using `classifier_recordings.py`.
