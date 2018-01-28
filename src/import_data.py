## Jordan Dubchak, January 2018
##
## data downloaded from: http://archive.ics.uci.edu/ml/datasets/Heart+Disease
##
## This file downloads the heart disease data with 14 variables from the Cleveland, Hungary, Switzerland and (Long Beach) VA databases.
## This file saves each data file as a csv, stored in the data/raw_data folder. No cleaning has been attempted yet. 

## necessary imports
import numpy as np
import pandas as pd
import requests
import io

## hard code variable names; variable explanations stored in doc/variables
hd_cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","label"] 

## urls from where the data is to be downloaded 
urls = {"clev": "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
       "hung": "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
       "swiss": "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
       "lbva": "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"}

## download the files 
clev = requests.get(urls["clev"])
hung = requests.get(urls["hung"])
swiss = requests.get(urls["swiss"])
lbva = requests.get(urls["lbva"])

## convert the response objects to csv files without headers
clev = pd.read_csv(io.StringIO(clev.content.decode('utf-8')), header = None)
hung = pd.read_csv(io.StringIO(hung.content.decode('utf-8')), header = None)
swiss = pd.read_csv(io.StringIO(swiss.content.decode('utf-8')), header = None)
lbva = pd.read_csv(io.StringIO(lbva.content.decode('utf-8')), header = None)

## add the hard-coded headers to each data frame
for df in [clev, hung, swiss, lbva]:
    df.columns = hd_cols

## save csv files to the data/raw_data folder
clev.to_csv("data/raw_data/cleveland.csv")
hung.to_csv("data/raw_data/hungary.csv")
swiss.to_csv("data/raw_data/switzerland.csv")
lbva.to_csv("data/raw_data/longbeachva.csv")