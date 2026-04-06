# 🍷 Clustering and Fitting Analysis of White Wine Quality

## 📌 Overview
This project performs a comprehensive data analysis on a White Wine dataset to explore how physicochemical properties influence wine quality. The analysis includes:

- Exploratory Data Analysis (EDA)
- Statistical analysis (mean, standard deviation, skewness, kurtosis)
- Data visualisation (categorical, relational, and statistical plots)
- K-Means clustering
- Linear regression (curve fitting)
- Model-based predictions

---

## 📂 Dataset Description
The dataset used in this project is the **White Wine Quality Dataset**, obtained from Kaggle:

🔗 https://www.kaggle.com/datasets/rohanpurohit0705/cleaned-white-dataset

### Features:
The dataset contains physicochemical properties of white wine samples:
- Fixed_Acidity
- Volatile_Acidity
- Citric_Acid
- Residual_Sugar
- Chlorides
- Free_SO2
- Total_SO2
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

### Target Variable:
- **Quality**: A rating score `(typically between 3-9)` representing wine quality based on sensory evaluation.

---

## 🧪 Methods and Techniques

### 1. Data Preprocessing
- Handling missing values
- Data type validation
- Basic statistical summaries

---

### 2. Visualisation

The following plots were generated:
- **Categorical Plot**: Wine counts by quality
- **Relational Plot**: Scatter plot of Alcohol vs Quality
- **Statistical Plot**: Correlation heatmap (selected features)
- **Clustering Plot**: Alcohol vs Sulphates with cluster centres
- **Elbow Plot**: Optimal cluster selection
- **Fitting Plot**: Linear regression (Alcohol vs Quality)

---

### 3. Statistical Analysis
Performed on the Alcohol variable:
- Mean = **10.59** 
- Standard Deviation = **1.22**
- Skewness = **0.45** `(slightly positively skewed)`
- Excess Kurtosis = **-0.70** `(Platykurtic)`

---

### 4. Clustering (K-Means)
- Features used: **Alcohol** and **Sulphates**
- Data was standardised before clustering
- Optimal clusters determined using:
  - Silhouette Score
  - Elbow Method (as confirmation)

✔ Best number of clusters: **3**
Clusters revealed distinct wine groupings based on chemical composition.

---

### 5. Fitting (Linear Regression)
- Feature: Alcohol  
- Target: Quality  

Model: `y = ax + b`
where y = quality, x = alcohol

Results:
- `a = 0.34 ± 0.01` 
- `b = 2.27 ± 0.11`

✔ Indicates a positive relationship between alcohol content and wine quality.

---

### 6. Predictions

#### Clustering:
white wine samples were categorized into 3 different closest cluster according to their alcohol and sulphates concentrations.

#### Fitting:
Quality predictions were made based on alcohol levels using the fitted model.

`Quality = 0.34*Alcohol + 2.27`

---

## Conclusion
This project showed that some chemical properties, especially alcohol, have a noticeable relationship with wine quality. The clustering also revealed that wines can be grouped into distinct categories based on alcohol and sulphates, rather than being randomly distributed. While the linear model is simple, it still gives a reasonable estimate of quality and highlights the general trend found in the data.
