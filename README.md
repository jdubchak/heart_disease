# Heart Disease Classification
Feature and Model Selection Mini Project for DSCI 753

## Purpose
The purpose of this mini project is to explore various forms of feature selection before building a model to predict whether an individual has heart disease. 

## Data
The data was collected in the late 1980's and was downloaded from the [UCI ML Repository](http://archive.ics.uci.edu/ml/datasets/heart+Disease). The principal investigators repsonsible for collecting the original data include:

 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

 ## Feature Descriptions 
 | Feature |  Feature Description  |
 |---------|-----------------------|
 | age | Age|
 | sex | Sex |
 | cp | Chest Pain Type |
 | trestbps| Resting Blood Pressure|
 | chol |Cholestoral |
 | fbs | Fasting Blood Sugar > 120mg/dl?|
 | restecg | Resting ECG Results| 
 |thalach |Maximum Heart Rate | 
 |exang | Did exercise induce the heart attack?| 
 | oldpeak|ST depression |
 |label | Diagnosis of Heart Disease|
 | location | Location of Data Collection (derived variable)|  
 
 ## Dependencies
 This analysis uses `scikit-learn v0.19.1`, `numpy v1.12.1`, `pandas v0.20.1`,
`matplotlib v2.0.2` and `seaborn v0.7.1`.
 
 ## Execution of this Analysis 
 To reproduce this analysis on your local machine, clone this repository and execute the driver script.
 
 `$ sh driver.sh`