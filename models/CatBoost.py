#CatBoost

from google.colab import drive
drive.mount('/content/drive')

# STEP 1: Install CatBoost 
!pip install catboost

# STEP 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.colors import ListedColormap

# STEP 3: Load Dataset
data_path = '/content/drive/MyDrive/TrafficData/Minnesota_TrafficData_2020-24.csv'  # Update your path if needed
df = pd.read_csv(data_path)
df.dropna(how='all', inplace=True)

# STEP 4: Convert wide format to long format
id_vars = ['station_id', 'dir_of_travel', 'lane_of_travel', 'date']
value_vars = [str(i) for i in range(1, 25)]  # columns '1' to '24'

hourly_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                    var_name='hour', value_name='volume')

# STEP 5: Extract datetime features
hourly_df['datetime'] = pd.to_datetime(hourly_df['date']) + pd.to_timedelta(hourly_df['hour'].astype(int) - 1, unit='h')
hourly_df['hour'] = hourly_df['datetime'].dt.hour
hourly_df['weekday'] = hourly_df['datetime'].dt.weekday
hourly_df['month'] = hourly_df['datetime'].dt.month
hourly_df.dropna(subset=['volume'], inplace=True)

# STEP 6: Prepare Features
features = hourly_df[['hour', 'weekday', 'month']]
target = hourly_df['volume']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# STEP 7: Train CatBoost Regressor
cat_model = CatBoostRegressor(verbose=0)
cat_model.fit(X_train, y_train)

# STEP 8: Predict and Evaluate
y_pred = cat_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(" CatBoost RMSE:", rmse)

# CatBoost RMSE: 363.472906937069 (you will get this output, no need to run this line)

# STEP 9: Prediction Plot
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title("CatBoost - Actual vs Predicted Traffic Volume")
plt.xlabel("Sample")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 10: Create heatmap classification DataFrame
heatmap_df = pd.DataFrame({
    'Predicted': y_pred,
    'hour': X_test['hour'].values,
    'weekday': X_test['weekday'].values
})

# STEP 11: Classify traffic levels
def classify_traffic(volume):
    if volume < 300:
        return 'Low'
    elif volume < 700:
        return 'Medium'
    else:
        return 'High'

heatmap_df['Traffic_Level'] = heatmap_df['Predicted'].apply(classify_traffic)

# Step 12: Create pivot table for heatmap
pivot_class = heatmap_df.pivot_table(
    values='Traffic_Level',
    index='hour',
    columns='weekday',
    aggfunc=lambda x: x.mode()[0] if not x.mode().empty else 'Low'
)

# STEP 13: Mapping classes to numeric values for color encoding
class_to_num = {'Low': 0, 'Medium': 1, 'High': 2}
pivot_numeric = pivot_class.replace(class_to_num)

# STEP 14: Define color map for numeric encoding
cmap = ListedColormap(['#56B4E9', '#F0E442', '#D55E00'])  # Blue, Yellow, Red

# STEP 15: Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_numeric, cmap=cmap, annot=pivot_class, fmt='', cbar=False)
plt.title(" CatBoost - Traffic Intensity by Hour & Weekday")
plt.xlabel("Weekday (0 = Monday)")
plt.ylabel("Hour")
plt.tight_layout()
plt.show()

