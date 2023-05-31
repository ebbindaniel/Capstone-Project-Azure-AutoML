
# Azure Machine Learning Engineer Capstone Project - MPG Prediction Using Azure ML

This project is aimed to train various models using Azure's Auto ML and Hyperdrive features and find the best model to be able to be deployed as a web service with REST API having active endpoints capable of predicting the miles per gallon for any vehicle given the details such as the Model Year, Number of cylinders, horsepower ,displacement,etc 



## Project Set Up and Installation

## Dataset

### Overview

**Context**

The data is technical spec of cars. The dataset is an external dataset from kaggle which can be found <a href='https://www.kaggle.com/datasets/uciml/autompg-dataset'>here </a> 

**Sources:**

(a) Origin: This dataset was taken from the StatLib library which is
maintained at Carnegie Mellon University. The dataset was
used in the 1983 American Statistical Association Exposition.

**Number of Attributes:** 7 


**Attribute Information:**

- mpg: continuous
- cylinders: multi-valued discrete
- displacement: continuous
- horsepower: continuous
- weight: continuous
- acceleration: continuous
- model year: multi-valued discrete


### Task


With this dataset I will be training multiple models for be able to predict the miles per gallons of any cars given the various technical details such as cylinders,displacement,horsepower,weight, acceleration,model year,etc are provided. 


### Access

I will be manually uploading the dataset to Azure ML studio from the data section as a tabular dataset in the provided blob storage via the account which is able to be accessed via the UI or the azure sdk given the workspace and resource details. 

![data1](screenshots/dataset.png?raw=true "data1")

![data2](screenshots/dataset1.png?raw=true "data2")

The data can be consumed via the SDK as shown as:

![data3](screenshots/dataset2.png?raw=true "data3")



## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment



### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
