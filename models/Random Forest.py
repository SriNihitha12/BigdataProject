#Random Forest

# STEP 1: Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install additional libraries (if needed)
!pip install pandas openpyxl

# Step 3: Import Random Forest Regressor from sklearn
from sklearn.ensemble import RandomForestRegressor

# Step 4: Create a Random Forest model with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Step 5: Train the model using training data
rf.fit(X_train, y_train)

# Step 6: Predict traffic volume using the test data
y_pred_rf = rf.predict(X_test)

# Step 7: Evaluate and print RMSE for Random Forest model
print("RMSE (Random Forest):", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# RMSE (Random Forest):363.7046963440941 (you will get this output, no need to run this line)

# Step 8: Create a pivot table to calculate average actual traffic volume by hour and weekday
pivot = hourly_df.pivot_table(values='volume', index='hour', columns='weekday', aggfunc='mean')

# Step 9: Set up the heatmap figure
plt.figure(figsize=(10, 6))
plt.title("Average Traffic Volume by Hour & Weekday")

# Step 10: Import seaborn and create the heatmap
import seaborn as sns
sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt='.0f')  # shows average volume per hour & weekday

# Step 11: Label the axes
plt.xlabel("Weekday (0 = Monday)")
plt.ylabel("Hour")

# Step 12: Display the plot
plt.show()
