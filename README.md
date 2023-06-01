
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

I will be manually uploading the dataset to Azure ML studio from the data section as a tabular dataset in the provided blob storage associated to the azure account which is able to be accessed via the UI or the azure sdk given the workspace and resource details. 

![data1](Screenshots/dataset.png?raw=true "data1")

![data2](Screenshots/dataset1.png?raw=true "data2")

The data can be consumed via the SDK as shown as:

![data3](Screenshots/dataset2.png?raw=true "data3")



## Automated ML

The following were the configuration for the AutoML run:

```
# automl settings:
automl_settings = {
    "n_cross_validations": 3,
    "primary_metric": "r2_score",
    "enable_early_stopping": True,
    "max_concurrent_iterations": 5,
    "experiment_timeout_minutes": 30
   
}

# automl config parameters:
automl_config = AutoMLConfig(
    task="regression",
    compute_target=compute_target,
    training_data=train_data,
    label_column_name=label,
    **automl_settings,
)
```
_experiment timeout minutes=30_

The default experiment was set to 30 mins. 

_task='regression'_

This defines the experiment type which in this case is Regression.

_primary metric='r2_score'_

r2_score was chosen as the primary metric for this regression problem.

_training data and label column name_ 

Here the Main dataset was was provided as training data and the label column was the Predictor column  - mpg (miles per gallon)


_n_cross_validations=3_

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). In this case i choose to have 3 Cross Validation subsets to reduce any overfitting as the metrics would be the average of 3 subset outputs generated. 

_enable_early_stopping = Enabled_

Early stopping is enabled to save compute costs if performance does not get better with each iteration. 



### Results

The results for the best model is as follows:

| AutoML Best Run Model | |
| :---: | :---: |
| id | AutoML_627bbbc0-c27d-4488-b9ed-663a5abaa043 |
| R2_Score | 0.8582 |
| Algortithm | VotingEnsemble |

The best model was the Voting Ensamble model with the r2 score of 0.8582


![aml1](Screenshots/automl_main.png?raw=true "aml1")

**The best model details are:**

-voting ensamble details


![aml2](Screenshots/automl_bestrun.png?raw=true "aml2")

-ensamble details - In this case Standard Scalar Wrapper, Light GBM Ensamble was used


![aml3](Screenshots/automl_bestmodel_ensemble.png?raw=true "aml3")


**Run Details**


![aml4](Screenshots/automl_run_details.png?raw=true "aml4")

**R2_Score Chart - Run Details**


![aml5](Screenshots/automl_r2_score.png?raw=true "aml5")

**AutoML Jobs**


![aml6](Screenshots/automl_job.png?raw=true "aml6")


**AutoML Experiments**


![aml7](Screenshots/automl_experiment.png?raw=true "aml7")


**Registered Model**


The Best Model was Registered as **AutoML_best_run**


![aml8](Screenshots/automl_registered_model.png?raw=true "aml8")

**Performance of different Models**

In our case the Voting Ensamble Model performed the best compared to other models as it had a better R2 Score as the model fit the data better than other models. There could be varaiety of factors that can affect performance such as batch size, learning rate, dropout,regularization methods, etc which can effect the performace.

There could be other factors such as variance and corelation in the dataset which effects some models but other models have weights and techniques to get by it delivering better performance. 

**Improvement**

I could have further improved it by testing various values for cross validations and also using feature selection to improve the performance.


## Hyperparameter Tuning

For the Hyperdrive Experiment i choose a **Random Forest Regressor Model**. The model is a simple yet effective model with good prediction results for a regression problem. 

Various hyperparameters affect the model performace so the various hyperparameters used are:

- **n_estimators:** number of trees in the foreset
- **max_depth:** max number of levels in each decision tree
- **bootstrap:** method for sampling data points (with or without replacement)

The various ranges used for parameter sampling are:

```
ps = RandomParameterSampling(
    {
        '--n_estimators' : choice(200, 300, 400),
        '--max_depth': choice(10, 20, 30, 40, 50),
        '--bootstrap': choice(1,0)
    }
)

```
The Policy used for early termination is the Bandit Policy which saves compute costs and time. 

```
# Bandit policy for early termination
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

```
The primary metric used here is **R2_Score**. The models are evaluated based on the highest R2_score. 


**Run_Details**


![hd1](Screenshots/hyperdrive_rundetails1.png?raw=true "hd1")

**R2_Score Run Details**


![hd2](Screenshots/hyperdrive_r2_score.png?raw=true "hd2")


![hd3](Screenshots/hyperdrive_r2_score2.png?raw=true "hd3")


**Hyperdrive Experiment**


![hd4](Screenshots/hyperdrive_r2_experiment.png?raw=true "hd4")

**Hyperdrive Jobs**


![hd5](Screenshots/hyperdrive_r2_jobs.png?raw=true "hd5")


**Main Hyperdrive Run with parameters**


![hd6](Screenshots/hyperdrive_main.png?raw=true "hd6")






### Results


| Hyperdrive Best Run Model | |
| :---: | :---: |
| id | HD_2743343d-e249-482d-b015-c37c75482dcf_11 |
| number of trees in the foreset | 300 |
| max number of levels in each decision tree | 20 |
| method for sampling data points (with or without replacement) | True |
| R2_Score | 0.9288 |
| Algortithm | Random Forest Regressor |

**Hyperdrive Best Run**


![hd7](Screenshots/hyperdrive_best_run.png?raw=true "hd7")


**Hyperdrive Registered Model**

![hd8](Screenshots/hyperdrive_registered_model.png?raw=true "hd8")


**Improvement**

The Hyperdrive model could have been improved be feature selection techniques and by using cross validation techniques. Also a parallel Ensamble model could add to the model performance. 

## Model Deployment

Based on both Auto ML and Hyperdrive models the best Hyperdive Model delivered a better R2_Score of 0.9288 and was used for model deployment.

The registered Hyperdrive model was deployed as a **ACI webservice**. The inference config was used to pass an entry script **score.py** which has the model init details to find the directory of the model and pass a input and output schema in the data section which is later used in consume section of the deployed webservice using REST API. 

Once the deployment was success the status is **healthy** with a successful deployment. 

![hd9](Screenshots/model_deploy_success.png?raw=true "hd9")


![hd10](Screenshots/model_deploy_active.png?raw=true "hd10")


The endpoints are tested using the REST API and a sample input data to obtain a predicted value.

![hd11](Screenshots/model_endpoint1.png?raw=true "hd11")

![hd11](Screenshots/model_endpoint2.png?raw=true "hd11")


## Screen Recording

The Link to the Screen Recording can be found <a href='https://youtu.be/aCott8YG-eQ'>here </a> 


