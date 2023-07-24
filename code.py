import pandas as pd
import numpy as np
df = pd.read_csv('Fish.csv')
df = df.drop('Species', axis=1)
y = df.Weight
X = df.drop('Weight', axis=1)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 1)

#s_X_train = scaler.fit_transform(X_train)
#s_X_test = scaler.transform(X_test)
LR = LinearRegression()
LR.fit(X_train, y_train)

import pickle
pickle.dump(LR, open('regressor.pkl', 'wb'))
