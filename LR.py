import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.model_selection import train_test_split


df = pd.read_excel('epl_player_stats_24_25.xlsx')
df = df.drop(columns=['Player Name','Club','Nationality','Hit Woodwork','Offsides'])

df_new = df.copy()

scaler = MinMaxScaler()


df_new[df_new.columns] = scaler.fit_transform(df_new)


df_in = df_new.drop(columns=['Market Value (EUR)'])
df_out = df_new['Market Value (EUR)']

x_train, x_test, y_train, y_test = train_test_split(df_in,df_out,test_size=0.25,random_state=1)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_train)

r2 = r2_score(y_train,y_pred)

print(f"R2: {r2}")