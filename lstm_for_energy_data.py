import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('energy.csv', sep=';', parse_dates=True)

# str to datetime
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, format = 'mixed')

# we fill the missing energy values with zeros
df['consumption (kWh)'].fillna(0, inplace=True)
df['consumption (kWh)'] = df['consumption (kWh)'].astype(str).str.replace(",", ".")
# str to float
df['consumption (kWh)'] = df['consumption (kWh)'].astype(float)

print(df.info())

# daily consumption sum
grouped_daily_sum = df.groupby([df['datetime'].dt.date])['consumption (kWh)'].sum()

# clean the data (sometimes the measurement at 00.00 creates daily deviation from the avg) 
grouped_daily_sum = grouped_daily_sum.loc[(grouped_daily_sum >= grouped_daily_sum.quantile(0.025)) & (grouped_daily_sum <= grouped_daily_sum.quantile(0.975))]

temp_series = df.copy()
temp_series['temperature (Celsius)'] = temp_series['temperature (Celsius)'].astype(str).str.replace(",", ".").astype(float)
temp_series.dropna(subset = ['temperature (Celsius)'],inplace=True)

grouped_temp = temp_series.groupby([temp_series['datetime'].dt.date])['temperature (Celsius)'].mean()

grouped_temp.plot(figsize=(10,6))
plt.title("Daily Temperature Mean (Celcius)")
plt.xlabel("Date")
plt.ylabel("Daily Temperature")
plt.savefig('daily_temperature_mean.png', dpi = 400)

fig, ax = plt.subplots()
ax.set_title("Energy Consumption (kWh)")
ax.set_xlabel("Day Number")
ax.set_ylabel("Consumption")
sns.lineplot(df['consumption (kWh)'])
fig.savefig('line_plot.png', dpi = 400)

max_fig, max_ax = plt.subplots()
plt.gca().axes.yaxis.set_ticklabels([])
max_ax.set_title("Daily Maximum Energy Consumption (kWh)")
max_ax.set_xlabel("Date")
max_ax.set_ylabel("Maximum Consumption")
sns.lineplot(grouped_daily_sum, color = "green")
max_fig.savefig('daily_max_plot.png', dpi = 400)

# initialize the model 
my_learner = DeepModelTS(
data = grouped_daily_sum,
Y_var = 'consumption (kWh)',
lag = 15,
LSTM_layer_depth = 50,
epochs = 10,
batch_size = 256,
train_test_split = 0.15
)

model = my_learner.LSTModel()
# Defining the lag that we used for training of the model 
lag_model = 15 # Getting the last period
ts = d['DAYTON_MW'].tail(lag_model).values.tolist()# Creating the X matrix for the model
X, _ = deep_learner.create_X_Y(ts, lag=lag_model)# Getting the forecast
yhat = model.predict(X)
yhat = deep_learner.predict()# Constructing the forecast dataframe
fc = d.tail(len(yhat)).copy()
fc.reset_index(inplace=True)
fc['forecast'] = yhat# Ploting the forecasts
plt.figure(figsize=(12, 8))
for dtype in ['DAYTON_MW', 'forecast']:  plt.plot(
    'Datetime',
    dtype,
    data=fc,
    label=dtype,
    alpha=0.8
  )plt.legend()
plt.grid()
plt.show()

# I will be using a ready-made LSTM time series model
class DeepModelTS():
    """
    A class to create a deep time series model
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        Y_var: str,
        lag: int,
        LSTM_layer_depth: int, 
        epochs=10, 
        batch_size=256,
        train_test_split=0
    ):

        self.data = data 
        self.Y_var = Y_var 
        self.lag = lag 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        """
        A method to create X and Y matrix from a time series list for the training of 
        deep learning models 
        """
        X, Y = [], []

        if len(ts) - lag <= 0:
            X.append(ts)
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag])
                X.append(ts[i:(i + lag)])

        X, Y = np.array(X), np.array(Y)

        # Reshaping the X array to an LSTM input shape 
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, Y         

    def create_data_for_NN(
        self,
        use_last_n=None
        ):
        """
        A method to create data for the neural network model
        """
        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]

        # The X matrix will hold the lags of Y 
        X, Y = self.create_X_Y(y, self.lag)

        # Creating training and test sets 
        X_train = X
        X_test = []

        Y_train = Y
        Y_test = []

        if self.train_test_split > 0:
            index = round(len(X) * self.train_test_split)
            X_train = X[:(len(X) - index)]
            X_test = X[-index:]     
            
            Y_train = Y[:(len(X) - index)]
            Y_test = Y[-index:]

        return X_train, X_test, Y_train, Y_test

    def LSTModel(self):
        """
        A method to fit the LSTM model 
        """
        # Getting the data 
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()

        # Defining the model
        model = Sequential()
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.lag, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Defining the model parameter dict 
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': False
        }

        if self.train_test_split > 0:
            keras_dict.update({
                'validation_data': (X_test, Y_test)
            })

        # Fitting the model 
        model.fit(
            **keras_dict
        )

        # Saving the model to the class 
        self.model = model

        return model

    def predict(self) -> list:
        """
        A method to predict using the test data used in creating the class
        """
        yhat = []

        if(self.train_test_split > 0):
        
            # Getting the last n time series 
            _, X_test, _, _ = self.create_data_for_NN()        

            # Making the prediction list 
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat

    def predict_n_ahead(self, n_ahead: int):
        """
        A method to predict n time steps ahead
        """    
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.lag)        

        # Making the prediction list 
        yhat = []

        for _ in range(n_ahead):
            # Making the prediction
            fc = self.model.predict(X)
            yhat.append(fc)

            # Creating a new input matrix for forecasting
            X = np.append(X, fc)

            # Ommiting the first variable
            X = np.delete(X, 0)

            # Reshaping for the next iteration
            X = np.reshape(X, (1, len(X), 1))

        return yhat    
