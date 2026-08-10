import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_excel('epl_player_stats_24_25.xlsx')
df = df.drop(columns=['Player Name','Club','Nationality'])

df = df[df['Position_GKP'] == 0]
df = df.drop(columns=['Position_GKP'])
df = df.drop(columns=['Goals Conceded','xGoT Conceded','Own Goals','Saves','Saves %','Penalties Saved','Punches','High Claims','Goals Prevented'])

df_new = df.copy()

scaler = MinMaxScaler()
df_new[df_new.columns] = scaler.fit_transform(df_new)


df_in = df_new.drop(columns=['Market Value (EUR)'])
df_out = df_new['Market Value (EUR)']

x_train, x_test, y_train, y_test = train_test_split(df_in,df_out,test_size=0.25,random_state=1)

nn = MLPRegressor(random_state=1, max_iter=1000,activation='relu',hidden_layer_sizes=300)
nn.fit(x_train,y_train)

y_pred = nn.predict(x_test)

r2 = r2_score(y_test,y_pred)

mse = mean_squared_error(y_test,y_pred)

print(f"R^2 scode is: {r2} and the MSE is: {mse}")