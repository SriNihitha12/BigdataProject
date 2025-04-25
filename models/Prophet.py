#Prophet

from google.colab import drive
drive.mount('/content/drive')

# STEP 1: Install Prophet
!pip install prophet --quiet

# STEP 2: Import Libraries
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# STEP 3: Load the Dataset
df = pd.read_csv('/content/drive/MyDrive/TrafficData/Minnesota_TrafficData_2020-24.csv')
df.dropna(how='all', inplace=True)

# STEP 4: Convert Wide to Long
hourly_df = df.melt(
    id_vars=['station_id', 'dir_of_travel', 'lane_of_travel', 'date'],
    value_vars=[str(i) for i in range(1, 25)],
    var_name='hour',
    value_name='volume'
)

# STEP 5: Create Datetime Column
hourly_df['datetime'] = pd.to_datetime(hourly_df['date']) + pd.to_timedelta(hourly_df['hour'].astype(int) - 1, unit='h')
hourly_df.dropna(subset=['volume'], inplace=True)

# Extract hour and weekday for later heatmap classification
hourly_df['hour'] = hourly_df['datetime'].dt.hour
hourly_df['weekday'] = hourly_df['datetime'].dt.weekday

# STEP 6: Prepare Data for Prophet
prophet_df = hourly_df[['datetime', 'volume']].rename(columns={'datetime': 'ds', 'volume': 'y'})
prophet_df = prophet_df.groupby('ds').mean().reset_index()

# STEP 7: Fit the Prophet Model
model = Prophet()
model.fit(prophet_df)

# STEP 8: Forecast Future (48 hours)
future = model.make_future_dataframe(periods=48, freq='H')
forecast = model.predict(future)

# STEP 9: Plot Forecast
model.plot(forecast)
plt.title(" Prophet Traffic Volume Forecast")
plt.xlabel("Datetime")
plt.ylabel("Volume")
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Forecast Components
model.plot_components(forecast)
plt.tight_layout()
plt.show()

# STEP 10: Calculate RMSE
actual = prophet_df['y'].values
predicted = model.predict(prophet_df)['yhat'].values
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(" Prophet RMSE:", rmse)

# Prophet RMSE: 80.57827892806492 (you will get this output, no need to run this line)

# STEP 11: Merge back predictions to add hour and weekday
forecast_trimmed = forecast[['ds', 'yhat']].copy()
forecast_trimmed['hour'] = forecast_trimmed['ds'].dt.hour
forecast_trimmed['weekday'] = forecast_trimmed['ds'].dt.weekday

# STEP 12: Classify Volume Levels
def classify_traffic(v):
    if v < 250:
        return 'Low'
    elif v < 500:
        return 'Medium'
    else:
        return 'High'

forecast_trimmed['Traffic_Level'] = forecast_trimmed['yhat'].apply(classify_traffic)

# STEP 13: Pivot Table for Heatmap
pivot = forecast_trimmed.pivot_table(
    values='Traffic_Level',
    index='hour',
    columns='weekday',
    aggfunc=lambda x: x.mode()[0] if not x.mode().empty else 'Low'
)

# STEP 14: Convert Categories to Numeric for Plotting
traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}
pivot_numeric = pivot.replace(traffic_map)

# Color map for 'Low', 'Medium', 'High'
color_map = {'Low': '#9be7ff', 'Medium': '#ffc857', 'High': '#ff5c5c'}
custom_cmap = ListedColormap([color_map['Low'], color_map['Medium'], color_map['High']])

# STEP 15: Plot Classification Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    pivot_numeric,
    cmap=custom_cmap,
    annot=pivot,
    fmt='',
    cbar=False
)
plt.title(" Prophet - Traffic Intensity by Hour & Weekday")
plt.xlabel("Weekday (0 = Monday)")
plt.ylabel("Hour")
plt.tight_layout()
plt.show()

