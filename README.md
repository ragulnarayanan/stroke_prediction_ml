#  Stroke Classification: Predicting Stroke Risk with Machine Learning

This project applies machine learning to **predict stroke risk** using demographic and clinical health data.  
It includes **Exploratory Data Analysis (EDA)**, **data preprocessing**, **dimensionality reduction**, **class balancing**, and **model evaluation** to identify the best-performing classification model.

---

## 📁 Project Structure

```
stroke-prediction/
│
├── data/
│   └── Stroke_dataset.csv        # dataset used for training/evaluation
│
├── notebooks/
│   └── Stroke_Prediction.ipynb   # main Jupyter notebook (code and results)
│
├── reports/
│   └── final_report.pdf          # final project report (summary of results)
│
├── README.md
└── requirements.txt
```
## 📊 Dataset Overview

**File:** `Stroke_dataset.csv`  
**Description:** Clinical dataset containing demographic, lifestyle, and medical information used to predict stroke occurrence.

| Feature | Description |
|----------|--------------|
| gender | Male, Female, or Other |
| age | Patient age |
| hypertension | 0 = No, 1 = Yes |
| heart_disease | 0 = No, 1 = Yes |
| ever_married | Marital status |
| work_type | Employment category |
| Residence_type | Urban or Rural |
| avg_glucose_level | Average blood glucose level |
| bmi | Body Mass Index |
| smoking_status | Smoking habits |
| stroke | Target variable (1 = Stroke, 0 = No Stroke) |

---

## 🔍 Exploratory Data Analysis (EDA)

Extensive **EDA** was performed to understand feature relationships and their impact on stroke occurrence.

**Key Insights:**
- **Age** and **average glucose level** showed strong correlation with stroke risk.  
- **Hypertension** and **heart disease** are major contributing factors.  
- Identified missing values in **BMI** and handled them via imputation.  
- Detected strong **class imbalance**, with far fewer stroke cases.

**Visualizations:**
- Histograms and KDE plots for numerical feature distributions  
- Boxplots for outlier analysis  
- Correlation heatmap for feature relationships  
- Countplots for categorical feature distributions  
- Pairplots for multivariate trends  

---

## ⚙️ Modeling Workflow

### 1. Data Preprocessing
- Handled missing values and encoded categorical variables  
- Scaled numerical features for optimal model performance  
- Applied **oversampling** techniques to balance classes  

### 2. Feature Engineering
- Implemented **Principal Component Analysis (PCA)** to reduce dimensionality and remove multicollinearity  

### 3. Model Training
Trained and compared six supervised learning models:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- K-Nearest Neighbors (KNN)  
- **Support Vector Classifier (SVC)**  

### 4. Model Evaluation
Evaluated models using multiple metrics to assess predictive performance:
- **Accuracy**  
- **Validation Error**  
- **Sensitivity (Recall)**  
- **Specificity**  
- **Precision**  
- **F1-Score**  
- **ROC-AUC**

Performance was visualized using:
- **Confusion Matrix**  
- **ROC Curve**

---

## 🏆 Results Summary

- **Support Vector Classifier (SVC)** achieved the **best overall performance**, balancing sensitivity, specificity, and precision.  
- **Oversampling** significantly improved minority class (stroke) detection.  
- **PCA** reduced feature dimensionality while retaining over 95% of the data variance.  
- Key predictors identified through EDA: **age**, **bmi**, and **avg_glucose_level**.  

---

## 🧩 Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn  
- **Visualization**: Matplotlib, Seaborn  
- **Tools**: Jupyter Notebook, Git, GitHub  

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/stroke-classification.git
cd stroke-classification

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/Stroke_Prediction.ipynb
