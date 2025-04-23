
## Team Name - Data Seekers
## Team Members
### Sri Nihitha Mandadi 
### Shiva Pavan Kumar Chava 
### Rakesh Kusa 
### Bhargava Reddy Erugula


# 🚦 Real-Time Traffic Analysis and Prediction

This project is a machine learning-based solution for forecasting traffic volume using real-world transportation data from the Minnesota Department of Transportation. We applied a variety of regression models to predict hourly traffic flow based on historical data, time features, and sensor information.

---

## 📁 Dataset

We used the `Minnesota_TrafficData_2020-24.csv` dataset containing 5 years of hourly traffic data collected from highway sensors across Minnesota.
This dataset was created by combining multiple annual traffic reports (from 2020 to 2024) into a single Excel file. Each year's data was merged, cleaned, and standardized to form a unified dataset for model training and analysis.

 **Download the dataset** from this Google Drive location: [Download Traffic Data] https://drive.google.com/file/d/1edNypNS0nUer5LNe63ZzDP7bpaNlG8T6/view?usp=sharing

 **Official Dataset Reference:** [MnDOT Hourly Traffic Volume Reports](https://mndot.org/traffic/data/reports-hrvol-atr.html)

---

##  Key Features

### 1. Data Cleaning and Preparation
- Removed null and empty rows
- Converted `date` and `hour` into proper datetime format
- Extracted features: `hour`, `weekday`, `month`
- Transformed wide format (24 hourly columns) into a long time-series structure

### 2. Analytical Components

**Traffic Trend Analysis**
- Average traffic volume by hour of day
- Volume trends across weekdays and months

**Model Training and Prediction**
- Regression models used:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Ridge & Lasso
  - SVR & KNN *(not optimal for large datasets)*
- Evaluated using RMSE to measure prediction accuracy

### 3. Visualization
- Traffic volume over time
- Heatmap of average volume by hour and weekday
- Actual vs Predicted plots for each model

### 4. Recommendations
-  Best Models: Extra Trees and Ridge Regression (fast + accurate)
-  Time features like `hour` and `weekday` significantly boost prediction
-  Avoid SVR and KNN for large-scale data due to performance issues

---

## Technical Implementation
- Built using **Python 3.x**
- Libraries used:
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
- Executed in **Google Colab** for free cloud compute
- Generated CSV files and visual plots as outputs

---

## Usage

1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/SriNihitha12/BigdataProject
   cd BigdataProject
   ```
2. Download the dataset and place it in the project folder:
   `Minnesota_TrafficData_2020-24.csv`
3. Open the Jupyter Notebook or Google Colab link
4. Run the notebook cells in order to:
   - Clean and preprocess data
   - Train models
   - Visualize results

---

## Output Files
- `rmse_results.csv` → RMSE values for all models
- `predicted_results.csv` → Comparison of actual vs predicted values
- PNG plots → Time series and heatmaps of traffic volume

---

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

This analysis provides actionable insights for city planners and traffic analysts. The use of multiple models, data cleaning techniques, and visual exploration helped identify high-traffic hours and effective predictors for traffic forecasting.
