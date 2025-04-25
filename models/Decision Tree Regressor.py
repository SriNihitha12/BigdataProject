#Decision Tree Regressor

# STEP 1: Mount Google Drive to access your dataset
from google.colab import drive
drive.mount('/content/drive')

# STEP 2: Install required libraries
!pip install pandas matplotlib seaborn scikit-learn

# STEP 3: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# STEP 4: Load Dataset
data_path = '/content/drive/MyDrive/TrafficData/Minnesota_TrafficData_2020-24.csv'  # Update if needed
df = pd.read_csv(data_path)
df.dropna(how='all', inplace=True)

# STEP 5: Convert wide format (1-24 hours) to long format
hourly_df = df.melt(
    id_vars=['station_id', 'dir_of_travel', 'lane_of_travel', 'date'],
    value_vars=[str(i) for i in range(1, 25)],
    var_name='hour',
    value_name='volume'
)

# STEP 6: Generate datetime and extract features
hourly_df['datetime'] = pd.to_datetime(hourly_df['date']) + pd.to_timedelta(hourly_df['hour'].astype(int) - 1, unit='h')
hourly_df.dropna(subset=['volume'], inplace=True)
hourly_df['hour'] = hourly_df['datetime'].dt.hour
hourly_df['weekday'] = hourly_df['datetime'].dt.weekday
hourly_df['month'] = hourly_df['datetime'].dt.month

# STEP 7: Prepare input features and target
features = hourly_df[['hour', 'weekday', 'month']]
target = hourly_df['volume']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# STEP 8: Train Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# STEP 9: Predict and Evaluate
y_pred_dt = dt_model.predict(X_test)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
print("RMSE (Decision Tree Regressor):", rmse_dt)

#RMSE (Decision Tree Regressor): 363.48427900104355 (you will get this output, no need to run this line)

# STEP 10: Create pivot table for heatmap
dt_heatmap_df = X_test.copy()
dt_heatmap_df['Predicted'] = y_pred_dt
dt_pivot = dt_heatmap_df.pivot_table(
    values='Predicted',
    index='hour',
    columns='weekday',
    aggfunc='mean'
)

# STEP 11: Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(dt_pivot, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Predicted Volume'})
plt.title("Decision Tree: Avg Predicted Traffic Volume by Hour & Weekday")
plt.xlabel("Weekday (0 = Monday)")
plt.ylabel("Hour")
plt.tight_layout()
plt.show()
