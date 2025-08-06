# Image-caption-and-segmentation-Time-series-analysis
Identifying objects/scenes in images, assigning labels to pixels 
# Time series analyis 
Analyzing data over time to identify trends ,patterns seasonality, enabled forecasting and informed decision-making
#image captioning and segmentation


#image captioning and segmentation

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input

# Load VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract image features
def extract_features(image):
    image = tf.image.resize(image, (224, 224)) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features

# Define LSTM-based captioning model
image_input = Input(shape=(None, None, 512))
caption_input = Input(shape=(None,))
embedding_layer = Embedding(input_dim=5000, output_dim=256)(caption_input)
lstm_layer = LSTM(512)(embedding_layer)
output_layer = Dense(5000, activation='softmax')(lstm_layer)

captioning_model = Model(inputs=[image_input, caption_input], outputs=output_layer)
captioning_model.compile(optimizer='adam', loss='categorical_crossentropy')

print("Image Captioning Model created successfully!")




import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input

# Load VGG16 model (pre-trained on ImageNet)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features from VGG16
inputs = Input(shape=(224, 224, 3))
features = vgg16(inputs)

# Decoder (segmentation layers)
x = Conv2D(512, (3,3), activation='relu', padding='same')(features)
x = UpSampling2D((2,2))(x)
x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)  # Output segmentation mask

# Create model
segmentation_model = Model(inputs, x)

# Compile model
segmentation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Image Segmentation Model using VGG16 created successfully!")



# Generate dummy segmentation mask (binary example)
segmentation_mask = np.random.randint(0, 2, (224, 224))

# Count pixel distribution
segmented_pixels = np.sum(segmentation_mask)
non_segmented_pixels = segmentation_mask.size - segmented_pixels

# Plot Pie Chart
labels = ['Segmented Area', 'Non-Segmented Area']
sizes = [segmented_pixels, non_segmented_pixels]
colors = ['lightblue', 'gray']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
plt.title("Segmentation Area Distribution")
plt.show()

# bar chart
plt.figure(figsize=(8,6))
plt.hist(segmentation_mask.flatten(), bins=2, color='blue', edgecolor='black')
plt.xticks([0,1], ['Non-Segmented', 'Segmented'])
plt.ylabel("Pixel Count")
plt.title("Segmentation Pixel Frequency")
plt.show()

#linechart
plt.figure(figsize=(10,4))
plt.plot(np.sum(segmentation_mask, axis=1), color='green', marker='o')
plt.xlabel("Row Index")
plt.ylabel("Number of Segmented Pixels")
plt.title("Segmentation Pattern Across Image Rows")
plt.grid()
plt.show()

# Bubble chart
x_coords = np.random.randint(0, 224, 50)
y_coords = np.random.randint(0, 224, 50)
sizes = np.random.randint(10, 200, 50)

plt.figure(figsize=(8,6))
plt.scatter(x_coords, y_coords, s=sizes, alpha=0.5, color='purple')
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Bubble Chart - Segmented Pixel Density")
plt.show()

#Histogram Chart
import numpy as np
import matplotlib.pyplot as plt

# Generate dummy segmentation mask (binary example)
segmentation_mask = np.random.randint(0, 2, (224, 224))  # 0 = background, 1 = segmented object

# Flatten the mask for histogram representation
pixel_values = segmentation_mask.flatten()

# Create Histogram
plt.figure(figsize=(8,6))
plt.hist(pixel_values, bins=[-0.5, 0.5, 1.5], color='blue', edgecolor='black', rwidth=0.8)

# Labels and Formatting
plt.xticks([0,1], ['Non-Segmented', 'Segmented'])
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Histogram of Segmented Pixels")
plt.grid(axis='y')

# Show the plot
plt.show()

# Scatter Plot
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 50]

# Create scatter plot
plt.scatter(x, y, color='blue', label='Data Points')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Scatter Plot')

# Show legend
plt.legend()

# Display the plot
plt.show()

#Area chart
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 30, 25]

# Create an area chart
plt.fill_between(x, y, color="skyblue", alpha=0.5)

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Area Chart")

# Display the plot
plt.show()

#box plot

import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = [7, 8, 9, 10, 15, 20, 22, 23, 24, 25, 30, 32, 35, 40, 45]

# Create a box plot
plt.boxplot(data, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))

# Add labels and title
plt.xlabel("Data")
plt.ylabel("Values")
plt.title("Simple Box Plot")

# Display the plot
plt.show()

#Heatmap

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data: 5x5 matrix
data = np.random.rand(5, 5)

# Create a heatmap
sns.heatmap(data, annot=True, cmap="coolwarm")

# Add title
plt.title("Simple Heatmap")

# Display the plot
plt.show()

#Radar Chart
import numpy as np
import matplotlib.pyplot as plt

# Sample data
categories = ["Speed", "Agility", "Strength", "Endurance", "Flexibility"]
values = [80, 60, 75, 90, 85]

# Convert categorical data to radians
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

# Complete the loop
values += values[:1]
angles = np.append(angles, angles[0])

# Create plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.fill(angles, values, color="skyblue", alpha=0.4)
ax.plot(angles, values, color="blue", linewidth=2)

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Display the radar chart
plt.title("Simple Radar Chart")
plt.show()

# Time series
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class StockTimeSeriesAnalysis:
    def __init__(self, ticker='AAPL', start_date='2022-01-01', end_date='2023-01-01'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.fetch_stock_data()

    def fetch_stock_data(self):
        """Fetch stock data using yfinance"""
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return stock_data['Close']

    def prophet_forecast(self):
        """Prophet Forecasting"""
        df = pd.DataFrame({
            'ds': self.data.index,
            # Explicitly flatten the values to ensure it's a 1-dimensional array
            'y': self.data.values.flatten()
        })

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        model.plot(forecast)
        plt.title(f'{self.ticker} Prophet Forecast')
        plt.show()
        return forecast

    def arima_forecast(self, order=(1,1,1)):
        """ARIMA Forecasting"""
        model = ARIMA(self.data, order=order)
        results = model.fit()
        forecast = results.forecast(steps=30)

        plt.figure(figsize=(10,6))
        # Ensure the index matches the forecast length for plotting
        plt.plot(self.data.index[-len(forecast):], forecast, label='ARIMA Forecast')
        plt.title(f'{self.ticker} ARIMA Forecast')
        plt.show()
        return forecast

    def lstm_forecast(self, look_back=60):
        """LSTM Neural Network Forecasting"""
        # Prepare data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data.values.reshape(-1, 1))

        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Split train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train model
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        plt.figure(figsize=(10,6))
        # Correcting the x-axis for plotting actual values to match the length of y_test
        plt.plot(self.data.index[train_size + look_back:train_size + look_back + len(y_test)],
                 scaler.inverse_transform(y_test.reshape(-1, 1)),
                 label='Actual')
        # Correct the x-axis for plotting predicted values to match the length of predictions
        plt.plot(self.data.index[train_size + look_back:],
                 predictions,
                 label='Predicted')
        plt.title(f'{self.ticker} LSTM Forecast')
        plt.legend()
        plt.show()

        return predictions

# Main execution
def main():
    # Create analysis instance
    analysis = StockTimeSeriesAnalysis(
        ticker='AAPL',
        start_date='2022-01-01',
        end_date='2023-01-01'
    )

    # Run forecasting models
    print("Prophet Forecast:")
    prophet_forecast = analysis.prophet_forecast()

    print("\nARIMA Forecast:")
    arima_forecast = analysis.arima_forecast()

    print("\nLSTM Forecast:")
    lstm_forecast = analysis.lstm_forecast()

if __name__ == "__main__":
    main()


class SARIMAForecasting:
    def __init__(self, data, seasonal_period=12):
        """
        Initialize SARIMA Forecasting Class

        Parameters:
        - data: Time series data
        - seasonal_period: Seasonal periodicity (default 12 for monthly data)
        """
        self.data = data
        self.seasonal_period = seasonal_period

    def check_stationarity(self):
        """
        Check time series stationarity using Augmented Dickey-Fuller test
        """
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(self.data)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])

        if result[1] <= 0.05:
            print("Data is stationary")
        else:
            print("Data is non-stationary, consider differencing")

    def difference_series(self, difference_order=1):
        """
        Apply differencing to make series stationary

        Parameters:
        - difference_order: Number of differences to apply
        """
        differenced_data = self.data

        for _ in range(difference_order):
            differenced_data = differenced_data.diff().dropna()

        return differenced_data

    def plot_acf_pacf(self):
        """
        Plot Autocorrelation and Partial Autocorrelation Functions
        """
        # Import necessary plotting functions
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        plot_acf(self.data, ax=ax1)
        plot_pacf(self.data, ax=ax2)

        plt.tight_layout()
        plt.show()

    def fit_sarima_model(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """
        Fit SARIMA Model

        Parameters:
        - order: (p,d,q) - Non-seasonal parameters
        - seasonal_order: (P,D,Q,m) - Seasonal parameters
        """
        try:
            model = SARIMAX(
                self.data,
                order=order,
                seasonal_order=seasonal_order
            )

            results = model.fit()
            print(results.summary())

            return results

        except Exception as e:
            print(f"Error fitting SARIMA model: {e}")
            return None

    def forecast(self, model, steps=12):
        """
        Generate SARIMA Forecast

        Parameters:
        - model: Fitted SARIMA model
        - steps: Number of forecast periods
        """
        try:
            forecast = model.get_forecast(steps=steps)
            forecast_mean = forecast.predicted_mean
            forecast_conf_int = forecast.conf_int()

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(self.data.index, self.data, label='Original Data')
            plt.plot(forecast_mean.index, forecast_mean, color='red', label='Forecast')
            plt.fill_between(
                forecast_conf_int.index,
                forecast_conf_int.iloc[:, 0],
                forecast_conf_int.iloc[:, 1],
                color='pink',
                alpha=0.3
            )
            plt.title('SARIMA Forecast')
            plt.legend()
            plt.show()

            return forecast_mean, forecast_conf_int

        except Exception as e:
            print(f"Forecast error: {e}")
            return None

    def model_evaluation(self, actual, forecast):
        """
        Evaluate model performance

        Parameters:
        - actual: Actual time series values
        - forecast: Predicted values
        """
        # Import mean_absolute_error
        from sklearn.metrics import mean_absolute_error

        mse = mean_squared_error(actual, forecast)
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mse)

        print("Model Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

def main():
    # Example usage with sample data
    # Replace with your actual time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
    data = pd.Series(
        np.cumsum(np.random.randn(len(dates))) + 10,
        index=dates
    )

    # Initialize SARIMA Forecasting
    sarima_forecast = SARIMAForecasting(data)

    # Check stationarity
    sarima_forecast.check_stationarity()

    # Plot ACF and PACF
    sarima_forecast.plot_acf_pacf()

    # Fit SARIMA Model
    model = sarima_forecast.fit_sarima_model(
        order=(1,1,1),           # Non-seasonal parameters
        seasonal_order=(1,1,1,12) # Seasonal parameters
    )

    # Generate Forecast
    if model:
        forecast_mean, forecast_conf_int = sarima_forecast.forecast(model, steps=12)

        # Model Evaluation
        sarima_forecast.model_evaluation(
            actual=data[-12:],    # Last 12 actual values
            forecast=forecast_mean
        )

if __name__ == "__main__":
    main()
