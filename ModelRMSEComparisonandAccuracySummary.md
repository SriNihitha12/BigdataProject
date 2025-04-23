This table summarizes the performance of different machine learning models used for traffic volume prediction.  
Each model was evaluated using RMSE (Root Mean Squared Error), where a lower value indicates better accuracy.


| **Model**           | **RMSE** | **Accurate or Not?**                                         |
|---------------------|---------:|---------------------------------------------------------------|
| Linear Regression   | 417.32   | Not the most accurate                                        |
| Decision Tree       | 363.70   | Accurate                                                     |
| Random Forest       | 363.70   | Accurate                                                     |
| XGBoost             | 363.70   | Accurate                                                     |
| CatBoost            | 363.47   | Most Accurate                                                |
| Prophet             | 80.58    | Highly Accurate (best for trend/seasonal time series)       |
