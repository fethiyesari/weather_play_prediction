# Weather Play Prediction

This project aims to predict play decisions based on weather conditions using linear regression.

## Dataset
The dataset is sourced from "weather_play_prediction.csv".

## Steps:
1. Encode categorical data (Outlook, Windy, Play).
2. Create DataFrames for encoded and numeric data.
3. Combine all DataFrames into a single DataFrame.
4. Split the data into training and test sets.
5. Train a linear regression model.
6. Predict play decisions on the test set.

## Usage
Run the `weather_play_prediction.py` file to execute the project.

## Dependencies
- pandas
- numpy
- scikit-learn

## FutureWork
Implement variable selection.
