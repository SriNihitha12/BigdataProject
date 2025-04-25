#XGBoost

from google.colab import drive
drive.mount('/content/drive')

# STEP1. Install XGBoost
!pip install xgboost

# STEP2. Import necessary libraries
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# STEP3. Create and train the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# STEP4. Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# STEP5. Evaluate the model using RMSE
xgb_mse = mean_squared_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(xgb_mse)

# Print RMSE
print(" XGBoost RMSE:", xgb_rmse)

# XGBoost RMSE: 363.6950130260243 (you will get this output, no need to run this line)

# STEP 6: Create a pivot table for heatmap visualization (Predicted volume)
predicted_pivot = pd.DataFrame(y_pred, columns=['Predicted'], index=X_test.index)
predicted_pivot['hour'] = X_test['hour']
predicted_pivot['weekday'] = X_test['weekday']
predicted_pivot = predicted_pivot.pivot_table(values='Predicted', index='hour', columns='weekday', aggfunc='mean')

#STEP 7: Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(predicted_pivot, cmap='coolwarm', annot=True, fmt='.0f', cbar_kws={'label': 'Predicted Volume'})
plt.title("Predicted Traffic Volume by Hour & Weekday")
plt.xlabel("Weekday (0 = Monday)")
plt.ylabel("Hour")
plt.tight_layout()
plt.show()
