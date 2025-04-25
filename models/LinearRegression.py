#Linear Regression

# STEP 1: Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

# STEP 2: Install necessary Python libraries (only needed once in Colab)
!pip install pandas matplotlib scikit-learn

# STEP 3: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# STEP 4: Define the path to the dataset file in Google Drive
data_path = '/content/drive/MyDrive/TrafficData/Minnesota_TrafficData_2020-24.csv'

# STEP 5: Load the traffic dataset using pandas
df = pd.read_csv(data_path)

# STEP 6: Display first few rows (for checking structure)
df.head()

# STEP 7: Remove any fully empty rows
df.dropna(how='all', inplace=True)

# STEP 8: Optional - Print column names to verify hour columns exist
print(df.columns)

# STEP 9: Reshape wide format (1–24 hours) into long format using melt
hourly_df = df.melt(
    id_vars=['station_id', 'dir_of_travel', 'lane_of_travel', 'date'],
    value_vars=[str(i) for i in range(1, 25)],
    var_name='hour',
    value_name='volume'
)

# STEP 10: Combine date and hour to create a full datetime column
hourly_df['datetime'] = pd.to_datetime(hourly_df['date']) + pd.to_timedelta(hourly_df['hour'].astype(int) - 1, unit='h')

# STEP 11: Sort data by datetime and reset the index
hourly_df.sort_values('datetime', inplace=True)
hourly_df.reset_index(drop=True, inplace=True)

# STEP 12: Drop rows where volume is missing
hourly_df = hourly_df[hourly_df['volume'].notnull()]

# STEP 13: Show structure of cleaned data
hourly_df.head()

# STEP 14: Plot a sample of traffic volume over time
plt.figure(figsize=(14, 4))
plt.plot(hourly_df['datetime'][:500], hourly_df['volume'][:500])
plt.title("Traffic Volume Over Time (Sample)")
plt.xlabel("Datetime")
plt.ylabel("Volume")
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 15: Extract time-related features: hour, weekday, and month
hourly_df['hour'] = hourly_df['datetime'].dt.hour
hourly_df['weekday'] = hourly_df['datetime'].dt.weekday
hourly_df['month'] = hourly_df['datetime'].dt.month

# STEP 16: Prepare input features and target variable
features = hourly_df[['hour', 'weekday', 'month']]
target = hourly_df['volume']

# STEP 17: Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# STEP 18: Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 19: Use the trained model to predict on the test data
y_pred = model.predict(X_test)

# STEP 20: Calculate and print evaluation metrics: MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("LinearRegression RMSE:", rmse)

#LinearRegression RMSE: 417.31918464462666 (you will get this output, no need to run this line)

# STEP 21: Plot a comparison of actual vs predicted traffic volumes (sample 100)
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title("Actual vs Predicted Traffic Volume")
plt.xlabel("Sample")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
