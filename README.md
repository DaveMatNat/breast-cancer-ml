# Breast Cancer Classification Using Machine Learning

This project utilizes machine learning techniques to classify breast tumors as malignant or benign based on the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/salihacur/breastcancerwisconsin). The goal is to develop a predictive model that can assist in the early detection of breast cancer.

## About the Dataset

The [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/salihacur/breastcancerwisconsin) comprises 700 instances of breast cancer cases, each with nine numerical features. The target variable (Class) indicates whether the tumor is benign (2) or malignant (4).

## Project Overview

The project involves the following steps:

1. **Data Preprocessing**:  
   Handling missing values, encoding categorical variables, and scaling features to prepare the data for modeling.

2. **Exploratory Data Analysis (EDA)**:  
   Analyzing the distribution of features and their relationships with the target variable to gain insights into the data.

3. **Model Development**:  
   To build predictive models, Implementing machine learning algorithms, including Decision Tree, KNN, SVM, Random Forest, NB, and Logistic Regression.

4. **Model Evaluation**:  
   Assessing the performance of the models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

5. **Feature Importance Analysis**:  
   Identifying the most significant features contributing to the model's predictions to understand the factors influencing breast cancer detection.

## Results

The models achieved high accuracy in classifying tumors with the following performance metrics:

- **Decision Tree**:  
  - Accuracy: 94%  
  - Precision: 94%  
  - Recall: 93%  
  - F1-score: 94%

- **K-nearest neighbors (KNN)**:  
  - Accuracy: 90%  
  - Precision: 92%  
  - Recall: 89%  
  - F1-score: 90%

- **Support Vector Machine (SVM)**:  
  - Accuracy: 91%  
  - Precision: 93%  
  - Recall: 89%  
  - F1-score: 91%

- **Random Forest**:  
  - Accuracy: 91%  
  - Precision: 93%  
  - Recall: 89%  
  - F1-score: 91%
    
- **Naïve Bayes (NB)**:  
  - Accuracy: 91%  
  - Precision: 93%  
  - Recall: 89%  
  - F1-score: 91%

- **Logistic Regression**:  
  - Accuracy: 89%  
  - Precision: 92%  
  - Recall: 87%  
  - F1-score: 89%
 
These results demonstrate the effectiveness of machine learning models in accurately detecting breast cancer based on tumor characteristics.

## Usage

To replicate the analysis:

### **Google Colab**: Link (**File -> Save a Copy in Drive**): [Google Colab](https://colab.research.google.com/drive/1EXmbVymfOSpfhuH-jA3DRNtIlsP8A4M1?usp=sharing)

### **Or**

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/DaveMatNat/breast-cancer-ml.git

   cd breast-cancer-ml
   ```
2. **Install the Required Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**:
   
> Launch Jupyter Notebook and open Breast_Cancer_ML.ipynb to explore the analysis and results.

---
