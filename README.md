Project Goal:
- Compare the performance of these three machine learning models before and after optimization: Logistic Regression (LR), Support Vector Classification (SVC), and Random Forest Classifier (RFC) for predicting customer churn.

PHASE 1: Modeling with Outliers Retained
- Data was cleaned for missing values and categorical variables were encoded.
- Hyperparameter tuning was performed using GridSearchCV.
- Despite tuning, the performance of all three models plateaued.

Results from PHASE 1:
- Logistic Regression (LR) : Train 80% of data score: 0.7027 || Test 20% of data score: 0.7040
- Support Vector Classification (SVC) : Train 80% of data score 0.4689 || Test 20% of data score: 0.4520
- Random Forest Classifier (RFC): Train 80% of data score: 1.00 || Test 20% of data score: 0.858

PHASE 1 Discussion:
- LR emerged as the best model. Although RFC had a great test score, it ended up overfitting. Further optimization was then carried out to improve model performance.

PHASE 2: Outlier Removal + Parameter Tunings
- Outliers were addressed using quantile-based filtering (1st–99th percentile) and later replaced with the median. Outliers accounted for 6.15% of the dataset.
- Outlier treatment was not applied to the binary columns to ensure the class distribution remained intact.
- Models were retrained and tuned again.

Results from PHASE 2:
- Logistic Regression (LR) : Train 80% of data score: 0.8715 || Test 20% of data score: 0.7845
- Support Vector Classification (SVC) : Train 80% of data score 0.7945 || Test 20% of data score: 0.8035
- Random Forest Classifier (RFC): Train 80% of data score: 0.9934 || Test 20% of data score: 0.858

PHASE 2 Discussion:
- Removing outliers and tuning parameters improved LR's performance but showed mild overfitting, significantly improved SVC's performance with no overfitting, and reduced RFC's performance.

Key Insights:
- Baseline performance doesn’t reflect the model’s true potential. Noise, outliers, and suboptimal hyperparameters can suppress its ability to learn patterns. Once outliers were addressed and hyperparameters were tuned, SVC could finally capture the structure in the data, drastically improving performance and even surpassing models that were initially better.
- Settling for the ‘best’ model in base modeling might not yield the best results after optimization, as proper outlier handling and hyperparameter tuning can transform underperforming models to outperform initially dominant ones while improving generalization.
