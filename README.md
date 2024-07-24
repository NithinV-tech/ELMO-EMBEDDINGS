# Assignment No. 4
# ELMO 

## Packages Required
->nltk
->pytorch
->Matplotlib


## How to run?
-> The code cannot be run on normal pc as it is compute heavy. It should be run on Kaggle Notebook
-> 2 files are there.
-> python Elmo.py
-> python Classification.py

## ASSUMPTIONS
-> The model has been put to train even during downstream classification task else it would give error as my lambdas and learnable function has been defined inside the Elmo class itself. 
-> However, i am ensuring that all the other parameters are freezed before running the downstream task.
-> i have written seperate freeze_parameters_except_lambda() and freeze_parameters_except_learnable_function inside the class and the same has been called while doing the downstream task.

## Link of saved model
->https://drive.google.com/drive/folders/1Mnwqm8Et23a7yczYSZB2AnBmFgzQ8UkL?usp=drive_link

-> I have uploaded all the pth files including those of hyperparameter tuning also.
