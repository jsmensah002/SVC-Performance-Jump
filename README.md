Project Goal:
- Compare the performance of these three machine learning models before and after optimization: Logistic Regression (LR), Support Vector Classification (SVC), and Random Forest Classifier (RFC) for predicting customer churn.

PHASE 1: Modeling with Outliers Retained
- Data was cleaned for missing values and categorical variables were encoded.
- Hyperparameter tuning was performed using GridSearchCV.
- Despite tuning, the performance of all three models plateaued.

Results from PHASE 1:
- Logistic Regression (LR) : Train 80% of data score: 0.7262 || Test 20% of data score: 0.733
- Support Vector Classification (SVC) : Train 80% of data score 0.4654 || Test 20% of data score: 0.4525
- Random Forest Classifier (RFC): Train 80% of data score: 1.00 || Test 20% of data score: 0.8600

PHASE 1 Discussion:
- LR emerged as the best model. Although RFC had a great test score, it ended up overfitting. Further optimization was then carried out to improve model performance.

PHASE 2: Outlier Removal + Parameter Tunings
- Outliers were addressed using quantile-based filtering (1st–99th percentile) and later replaced with the median. Removing outliers entirely was not possible because the target column is binary, containing only 1 (True) and 0 (False). This would have distorted the dataset.
- Models were retrained and tuned again.

Results from PHASE 2:
- Logistic Regression (LR) : Train 80% of data score: 0.8719 || Test 20% of data score: 0.785
- Support Vector Classification (SVC) : Train 80% of data score 0.7945 || Test 20% of data score: 0.785
- Random Forest Classifier (RFC): Train 80% of data score: 0.9922 || Test 20% of data score: 0.785

PHASE 2 Discussion:
- Removing outliers and tuning parameters improved LR's performance but showed mild overfitting, significantly improved SVC's performance with no overfitting, and reduced RFC's performance.

Key Insights:
- This highlights that models which perform poorly in baseline modeling can become the strongest predictors after removing noise and carefully tuning parameters. Proper outlier handling and hyperparameter tuning can transform underperforming models, sometimes outperforming initially dominant models while achieving better generalization.
