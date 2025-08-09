
# Autism Prediction using Machine Learning: A Classification Project

## 1. Project Overview

This project focuses on the **development of a machine learning model designed to predict autism** in individuals . Recognizing that there are **no certain diagnostic methods** currently available for this neurological disorder, this initiative explores the potential of machine learning to provide predictive insights . The comprehensive process, encompassing data acquisition, rigorous preprocessing, model training, and performance evaluation, was meticulously conducted using established data science methodologies.

## 2. Problem Statement

Autism is a neurological disorder profoundly affecting social interaction, eye contact, and often presenting with various behavioral issues. A significant challenge in this medical domain is the **absence of definitive diagnostic methods**, which impedes early and accurate identification . This project endeavors to address this critical gap by leveraging machine learning techniques to predict the likelihood of autism, thereby offering a data-driven approach to a complex diagnostic challenge.

## 3. Project Objective

The primary objective of this project is to **construct and rigorously evaluate a machine learning model capable of accurately predicting whether an individual is likely to have autism or not** . By achieving high predictive accuracy, this work aims to contribute a valuable data-driven tool where traditional diagnostic approaches are limited.

## 4. Technologies and Libraries Used

The project was primarily developed in **Python**, leveraging a comprehensive suite of industry-standard data science and machine learning libraries [5]:

*   **Pandas**: Utilized for robust data manipulation and analysis.
*   **NumPy**: Employed for fundamental numerical operations and array manipulation.
*   **Matplotlib / Seaborn**: Indispensable for comprehensive data visualization and Exploratory Data Analysis (EDA).
*   **Scikit-learn (Sklearn)**: Crucial for machine learning model implementation, data splitting, preprocessing (e.g., `StandardScaler`, `LabelEncoder`), and evaluation metrics .
*   **XGBoost**: Implemented specifically for the `XGBClassifier` model, a powerful gradient boosting framework .
*   **Imblearn**: Specifically, `RandomOverSampler` was used to effectively address and handle dataset imbalance, preventing skewed model performance .

## 5. Dataset

The dataset central to this project comprises **800 rows and 22 columns**, containing diverse attributes hypothesized to be relevant for autism prediction [9]. This dataset was sourced externally. Initial exploratory analyses indicated a mix of numerical and categorical features, and while explicit null values were not detected, ambiguous entries required specific handling .

## 6. Methodology

The project strictly adhered to a structured machine learning pipeline to ensure methodological rigor and reliable results :

### 6.1. Data Cleaning and Preprocessing

Raw data underwent significant preprocessing to ensure optimal quality and consistency for model training [10]:
*   **Handling Ambiguous Values**: Ambiguous categorical entries such as '?', 'others', and 'Others' were systematically replaced with a consistent 'Others' to eliminate discrepancies and standardize data .
*   **Binary Encoding**: Boolean values 'yes' and 'no' were converted to numerical '1' and '0' respectively, facilitating numerical processing by machine learning algorithms .
*   **Outlier Removal**: Outliers identified in the 'result' column (values less than -5) were carefully removed. This step resulted in a minimal loss of only two data points, deemed acceptable for maintaining the overall integrity and representativeness of the dataset .

### 6.2. Exploratory Data Analysis (EDA)

Comprehensive EDA was performed to gain profound insights into data characteristics, distributions, and potential predictive relationships [15]:
*   **Imbalance Assessment**: Pie charts were generated to visually confirm that the **dataset was significantly imbalanced**, heavily skewed towards the negative class. This critical finding underscored the necessity for specific strategies to address imbalance during modeling .
*   **Feature Distribution Analysis**: Count plots, distribution plots, and box plots were extensively utilized to visualize the distributions of both numerical and categorical features .
*   **Key Observations**:
    *   It was observed that lower scores on most indicators (with the notable exception of `A10_Score`) correlated strongly with a lower probability of autism .
    *   The `country_of_res` feature provided insights into autism prevalence, revealing that some countries exhibited approximately 50% autism cases within their available data, suggesting geographical relevance .
    *   Continuous data, specifically 'age' and 'result', displayed noticeable skewness, indicating a need for transformation.

### 6.3. Feature Engineering

New features were carefully derived, and existing ones transformed, to potentially enhance model performance and glean deeper insights [14]:
*   **Age Group Categorization**: An `ageGroup` feature was engineered by categorizing the `age` column into 'Toddler', 'Kid', 'Teenager', 'Young', and 'Senior' groups. Analysis of this new feature indicated that 'Young' and 'Toddler' groups showed comparatively lower chances of autism .
*   **Clinical Score Summation**: A `sum_score` feature was created by summing individual clinical scores from `A1_Score` to `A10_Score`. This aggregation revealed a significant correlation: **a higher `sum_score` strongly indicated a higher probability of autism**, while scores below 5 suggested a rare chance of autism .
*   **Log Transformation**: To address the positive skewness observed in the 'age' column, **log transformations were applied**. This successfully normalized its distribution, facilitating more effective model training .
*   **Highly Correlated Feature Identification**: A heatmap was generated to visualize feature correlations, enabling the identification and subsequent removal of highly correlated features (correlation > 0.8) before model training. This step prevents redundancy and improves the model's ability to learn distinct, useful patterns .

### 6.4. Model Training and Evaluation

The meticulously prepared dataset was subsequently utilized to train and rigorously evaluate a selection of prominent machine learning models :
*   **Data Splitting**: Features and target variables were clearly separated, and the data was then split into training and validation sets using an 80/20 ratio .
*   **Imbalance Handling**: The critical issue of dataset imbalance in the training data was addressed by employing the **Random Over Sampler** on the minority class. This technique ensures a balanced representation for robust model learning and improved predictive capability for the less frequent class .
*   **Data Normalization**: `StandardScaler` was applied to normalize the features, a crucial step for achieving stable and faster convergence during model training .
*   **Model Selection**: A curated selection of state-of-the-art machine learning classification models were trained and evaluated:
    *   **Logistic Regression**
    *   **XGBClassifier**
    *   **Support Vector Classifier (SVC)** with an 'rbf' kernel.
*   **Performance Assessment**: Models were compared based on their training and validation accuracy, specifically using `metrics.roc_auc_score` . **Logistic Regression and SVC were identified as the top-performing models** on the validation data, demonstrating strong generalization capabilities with minimal disparities between training and validation accuracy .

## 7. Project Impact

This project distinctly demonstrates the **practical and impactful application of machine learning in addressing a significant real-world problem within healthcare**, particularly in predicting a neurological disorder where conventional diagnostic methods are currently limited. The successful implementation of this predictive model underscores how data-driven approaches can offer invaluable insights and potentially facilitate earlier identification of autism, paving the way for timely intervention and support.
