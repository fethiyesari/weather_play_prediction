# Libraries
import pandas as pd
import numpy as np

# Upload data
data = pd.read_csv("Weather_Play_Prediction.csv")

# ENCODER

# Converting the Outlook column to numerical values
outlook = data.iloc[:,0].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
outlook_le = le.fit_transform(outlook) # each one is assigned a number

ohe = preprocessing.OneHotEncoder()
outlook_ohe = ohe.fit_transform(outlook_le.reshape(-1, 1)).toarray()

# Converting the Windy column to numerical values
windy = data.iloc[:,3].values

le = preprocessing.LabelEncoder()
windy_le = le.fit_transform(windy)

# Converting the Play column to numerical values
play = data.iloc[:,-1].values

le = preprocessing.LabelEncoder()
play_le = le.fit_transform(play)

# DATAFRAME

# Outlook_ohe
outlookDataFrame = pd.DataFrame(data = outlook_ohe, index=range(14), columns=["overcast", "rainy", "sunny"])

# Temperature
temperatureDataFrame = pd.DataFrame(data = data, index = range(14), columns=["temperature"])

# Humidity
humidityDataFrame = pd.DataFrame(data = data, index= range(14), columns=["humidity"])

# Windy_le
windyDataFrame = pd.DataFrame(data = windy_le, index= range(14), columns=["windy"])

# Play_le
playDataFrame = pd.DataFrame(data= play_le, index=range(14), columns=["play"])

'''
Independent variables -> outlook, temperature, humidity, windy
Dependent variable -> play
'''

# Combining all dataframes
combinedData = pd.concat([outlookDataFrame, temperatureDataFrame, humidityDataFrame, windyDataFrame, playDataFrame], axis=1) 

# Combining independent variables
independentVariablesDataFrame = pd.concat([outlookDataFrame, temperatureDataFrame, humidityDataFrame, windyDataFrame], axis=1)

# TRAIN AND TEST
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(independentVariablesDataFrame, playDataFrame, test_size=0.33, random_state=0)

# MODEL
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

'''
# Variable Selection
...
...
'''
