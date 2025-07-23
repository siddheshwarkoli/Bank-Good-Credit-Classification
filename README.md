# <b>PROJECT NAME : BANK GOOD CREDIT (PR-0015)
### <b>CONTRIBUTION - SIDDHESHWAR KOLI
## <B>PROJECT SUMMARY
<B>
    
     This project aims to predict which customers at Bank GoodCredit are at risk of having bad credit.
     By analyzing customer account, enquiry, and demographic data, the goal is to identify key factors that impact credit risk.
     The model will help the bank reduce bad debt and make better decisions about who to offer credit to.
     The model's performance will be measured using Gini and rank ordering.

## <B>PROBLEM STATEMENT

### <B>OBJECTIVE
    - The objective of this project is to develop a machine learning model that predicts whether a customer has good or bad credit history based on their account details, enquiry data, and demographics.

    - TARGET VARIABLE : Bad_label
        - 0 → Good Credit History (Customer pays on time)
        - 1 → Bad Credit History ((falls into 30 DPD + bucket)

    - Key Metric: Gini Coefficient (Measures how well the model differentiates good vs. bad customers)
    - Current Benchmark: Gini Score = 37.9 (We need to improve this)

### <B>WHY IS THIS IMPORTANT ?
    - Helps the bank make better lending decisions.
    - Reduces the risk of bad loans and financial losses.
    - Improves customer segmentation based on creditworthiness.


### ALGORITHMS USED FOR PROJECT
  - LOGISTIC REGRESSION
  - DECISION TREE CLASSIFIER
  - GRADIENT BOOSTING
  - KNN CLASIFIER
  - RANDOM FOREST CLASSIFIER

### MODEL COMPARISION OF GINI VALUE AND F1 SCORE OF TRAIN AND TEST DATA
| Index | Model                        | Train F1 Score | Test F1 Score | Gini Value |
| ----- | ---------------------------- | -------------- | ------------- | ---------- |
| 0     | Logistic Regression          | 52.76          | 52.96         | 0.074592   |
| 1     | Decision Tree Classifier     | 88.64          | 85.88         | 0.368231   |
| 2     | Gradient Boosting            | 96.56          | 92.91         | 0.863389   |
| 3     | KNN Classifier               | 84.68          | 88.24         | 0.736850   |
| 4     | Random Forest Classification | 100.00         | 99.71         | 0.994643   |




# <b>FINAL PROJECT REPORT
## <b>UNDERSTANDING THE PROBLEM
### <b>1) CHALLENGES

    - Clearly defining the goal was important to ensure we built the right model.

    - Understanding banking terms like "30 DPD+ bucket" was tricky.

    - We had to carefully classify customers as "good" or "bad" based on credit history.

    - Financial History: Credit balance, high credit amount, and payment history.

    - Demographic Information: Age, location, and customer ID.

    - Transaction Details: Account type, reporting date, and account opening date.

    - Target Variable: Bad_label (1 = Bad credit, 0 = Good credit)

### <b>2) UNDERSTANDING AND CHALLENGES IN DATASET

    - The dataset was large, so it took time to load.

    - We had to check for missing values and incorrect data types.

    - Some data entries were duplicated or wrongly classified.


### <b>3) CHECKING FOR MISSING VALUES

    - Identifying which columns had missing values.

    - Deciding how to handle missing data (filling with average values or removing rows).

### <b>4) CHECKING AND CONVERTING DATA-TYPE

    - Some numbers were stored as text, which could cause errors.

    - We had to convert categorical data into numerical.

### <b>5) DATA PREPROCESSING

#### <b>A) REMOVING DUPLICATE RECORDS
    - Finding and removing duplicate rows to avoid incorrect model training.
#### <B>B) EXPLORATORY DATA ANALYSIS (EDA)
    - Choosing the best graphs and charts to understand the data.
    - Used histograms to analyze the distribution of numerical
    - Used countplot for catagorical to analyse the distribution
    - Finding outliers but we didn’t handle outliers because some outliers were imp for this project
    - Understanding relationships between different factors affecting credit score.
#### <B>C) PREPARING DATA FOR THE MODELTRAINING
    - checked high correlation and remove above >90
    - Removed unwanted columns and merged data
    - In a Texas Employee Salary Prediction dataset, checking and handling outliers might not be important
    - The outliers represent valid data points (e.g., high salaries for senior staff or low salaries for part-time employees), and removing them would lead to a loss of valuable information
    - So we did not handle Outliers
    - Sacling the data is not required for the given dataset because we have only limited or very less variations in the feature columns.
#### <B>D) CHECKING FOR IMBALANCE
    - The dataset had more "good" credit cases than "bad" ones.
    - We had to balance the data using techniques like SMOTE or undersampling.

### <B>6) MODEL TRAINING
#### <B>A) SPLITING DATA INTA TRAINING AND TESTING SETS
    - Deciding how much data to use for training and testing.
    - The dataset was split into training (80%) and testing (20%) sets.
#### <B>B) CHOOSING AND TRAINING MODEL
    - Selecting the best model for eg. Logistic Regression, Decision Tree, Random Forest.
    - Some models gives overfitting
#### <B>C) CHECKING MODEL PERFORMANCE
    - Understanding how well the model worked using Accuracy, F1-score, and AUC-ROC.
#### <B>D) FINE TUNNING THE MODEL
    - Choosing the best settings for the model (like tree depth and learning rate).
    - Running multiple tests without making the model too complex.
    - Saving the Model for Future Use
#### <B>E) FINAL MODEL AND PERFORMANCE
    - Final Model Chosen: Random Forest (best accuracy & AUC-ROC score).
    - Evaluation: Balanced accuracy, reduced overfitting, and performed well on new data.

### <b>7) MODEL SUMMARY 
#### <B>A) Random Forest Classifier:
    - Highest Train F1 (100%) and Test F1 (99.71%).
    - Near-perfect Gini (0.9946).
    - Possible overfitting, though generalization seems strong based on test score.

#### <B>B) Gradient Boosting:
    - Very strong performance across the board: Train F1 (96.56%), Test F1 (92.91%), and Gini (0.8634).
    - Balanced and powerful model with good generalization.

#### <B>C) KNN Classifier:
    - Surprisingly strong Test F1 (88.24%) and Gini (0.7369).
    - Lower Train F1 (84.68%), which may indicate slight underfitting or good generalization.

#### <B>D) Decision Tree Classifier:
    - Moderate performance: Test F1 (85.88%), Gini (0.3682).
    - Likely overfitting (Train F1: 88.64% vs. lower Gini).

#### <B>E) Logistic Regression:
    - Weakest performance: F1 Scores ~52%, Gini only 0.0746.
    - Performs only slightly better than random — not suitable for this problem.

  ### <B>8) Final Conclusion:
    - Best Overall Model: Random Forest, due to top performance on both F1 score and Gini.
    - Most Balanced Alternative: Gradient Boosting, offering high accuracy with strong generalization and excellent rank ordering.
    - Avoid Using: Logistic Regression, due to poor classification and discrimination.
    - If model interpretability is important, Gradient Boosting (with tools like SHAP) could be preferred over Random Forest, which is more complex but slightly more powerful.

# <b>END OF PROJECT
# <b>THANK YOU
