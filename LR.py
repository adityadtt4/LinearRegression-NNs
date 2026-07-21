import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

df = pd.read_excel('player_data.xlsx')




df_new = df.copy()
df_new = df_new.drop(columns=['Player','National Team'])



df_out = df_new['Market Value (EUR)']
df_in = df_new.drop(columns=['Market Value (EUR)'])

x_training, x_test, y_training, y_test = train_test_split(df_in, df_out, test_size=0.25, train_size=0.75, random_state=1)

lr = LinearRegression()
lr.fit(x_training, y_training)

y_pred = lr.predict(x_test)

print(f"R2 Score: {r2_score(y_test, y_pred)}")







