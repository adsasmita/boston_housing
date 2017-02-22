# boston_housing

A machine learning project using [Scikit-learn](http://scikit-learn.org/stable/) Python library to predict housing market prices in [Boston, Massachusetts](https://en.wikipedia.org/wiki/Boston).

## Files

To open the main code, simply open [`boston_housing.ipynb`](https://github.com/adsasmita/boston_housing/blob/master/boston_housing.ipynb) on any desktop browser, or you can download and run the cells in a Python 2 environment. The code is presented in a [Jupyter Notebook](https://github.com/jupyter/notebook) / iPython notebook for readability purposes.

Visualization codes are contained in [`visuals.py`](https://github.com/adsasmita/boston_housing/blob/master/visuals.py)

## Overview

The Boston housing market is highly competitive and saturated. Suppose that we want to be the best real estate agents in the area. To compete with other agents, we decided to leverage a few basic machine learning concepts to determine the best selling price for their home. Luckily, we came across the Boston Housing dataset which contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. We decided to build an optimal model based on a statistical analysis with the tools available. The model will then be used to estimate the best selling price for our clients' homes.

This project will also evaluate the performance and predictive power of the model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value.

This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.

## Data

The dataset used in this project is included as `housing.csv`. The dataset originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts.

**Features**
1. CRIM: per capita crime rate by town 
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
3. INDUS: proportion of non-retail business acres per town 
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
5. NOX: nitric oxides concentration (parts per 10 million) 
6. RM: average number of rooms per dwelling 
7. AGE: proportion of owner-occupied units built prior to 1940 
8. DIS: weighted distances to five Boston employment centres 
9. RAD: index of accessibility to radial highways 
10. TAX: full-value property-tax rate per 10000 dollars
11. PTRATIO: pupil-teacher ratio by town 
12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
13. LSTAT: % lower status of the population 
14. MEDV: Median value of owner-occupied homes in 1000 dollars

## Dependencies

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [Scikit-learn](http://scikit-learn.org/stable/)
