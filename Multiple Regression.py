# -*- coding: utf-8 -*-
"""
Machine Learning - Multiple Regression

@author: anamc
"""

import pandas 
import pandas as pd
import numpy as np
import os
from sklearn import linear_model



# finding the current directory
abs_path = os.getcwd()
abs_path

# change to desired folder where .csv file is present - Use forward backslash
path = r'D:\Dropbox\Dropbox\A_CESEP Ana Ilie\Data Science AQ\Machine Learning'
data = pd.read_csv(path + '/data.csv')


# make a list of the independent values and call this variable X
# Put the dependent values in a variable called y

X = data[['Weight', 'Volume']]
y = data['CO2']

# From the sklearn module we will use the LinearRegression() method to create a linear regression object. 
# This object has a method called fit() that takes the independent and dependent values as parameters and 
# fills the regression object with data that describes the relationship:
    
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Now we have a regression object that are ready to predict CO2 values based on a car's weight and volume:

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)

#Print the coefficient values of the regression object and the result array represents the coefficient values of weight and volume
#These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g, and if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g

print(regr.coef_)




    
