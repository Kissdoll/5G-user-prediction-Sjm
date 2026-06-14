# 5G User Prediction Project  
Fujian Normal University, School of Software Engineering — AI Group Assignment  

## Problem Statement  
Predict whether a telecom user is a 5G subscriber based on demographic & behavioral features (e.g., tariff plan, data usage, regional info). This is a binary classification task targeting real-world operator marketing optimization.

## Dataset  
1. Source: Provided by course platform (`train.csv`)  
2. Features: 58 fields(categorical and numerical)  
3. Target: target (1=5G user, 0=non-5G user)  
4. Metric: AUC Score (higher = better)

## Implementation  
- Models: KNN + LogistRegression + DecissionTree + LightGBM (RandomForest + GBDT) +  SVM
- Tools: Python 3.14, scikit-learn, pandas, lightgbm, matplotlib,
- Key Steps:  
  1. Data Loading & Exploration (Load the train.csv dataset and perform initial exploratory data analysis (EDA) to understand the basic characteristics of the data)
  2. Data Preprocessing (Handle missing values and split the dataset into training and testing sets using train_test_split ) 
  3. Feature Engineering (Apply feature scaling to standardize the numerical features)  
  4. Model Training (Train multiple machine learning models and perform hyperparameter tuning using Grid Search (GridSearchCV))
  5. Evaluation (Calculate the AUC score to evaluate model performance and compare the effectiveness of the different models)

## Team & Roles  
- Class: Class 1, Software Engineering, Grade 2024, Fujian Normal University
- Members: Jiamo Song, Yuchen Weng, Mengcheng He, Shengjie He, Xinrui Lin, Zhitao Zhang  
- Roles: 
  Jiamo Song: 
    1. Data loading, preprocessing, and standardization; 
    2. Orchestrated model training with cross-validation and grid search; 
    3. Model evaluation and ROC curve synthesis; 
    4. Repository management, documentation, and final report compilation. 
  Yuchen Weng:
    1. KNN model development and hyperparameter optimization; 
    2. Contributed to the translation and formatting of the project documentation.
  Mengcheng He:
    1. Logistic Regression model development and hyperparameter optimization; 
    2. Contributed to the translation and formatting of the project documentation.
  Shengjie He:
    1. Decision Tree model development and hyperparameter optimization; 
    2. Enhanced source code readability through detailed English documentation and annotations.
  Xinrui Lin:
    1. Bagging model development; 
    2. Comparative analysis of Random Forest implementations (LightGBM vs scikit-learn); 
    3. Contributed to team coordination and self-assessment reporting.
  Zhitao Zhang
    1. Boosting model development; 
    2. Comparative analysis of GBDT implementations (LightGBM vs scikit-learn); 
    3. Contributed to the final laboratory report and document formatting.

