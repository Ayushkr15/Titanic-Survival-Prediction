# 🚢 Titanic Survival Prediction

> A comprehensive machine learning project predicting passenger survival on the Titanic using advanced feature engineering and model optimization techniques.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-red.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project demonstrates end-to-end machine learning workflow for the classic Titanic survival prediction challenge. Through comprehensive data preprocessing, feature engineering, and model evaluation, we achieve **81.6% accuracy** using ensemble of multiple algorithms.

### 🏆 Key Results
- **🥇 Best Model**: Logistic Regression with **81.6% accuracy** and **0.744 F1-score**
- **📊 Feature Discovery**: Sex, Age, and Fare identified as most discriminative factors
- **🔄 Model Consistency**: Strong agreement across three different algorithms
- **📈 Historical Accuracy**: Predictions align with documented survival patterns (60% non-survivors, 40% survivors)

## 📋 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🛠️ Installation & Setup](#️-installation--setup)
- [🏗️ Project Structure](#️-project-structure)
- [⚙️ Data Preprocessing](#️-data-preprocessing)
- [🤖 Models & Performance](#-models--performance)
- [📈 Key Insights](#-key-insights)
- [🚀 Usage](#-usage)
- [📁 Files Description](#-files-description)
- [🎨 Visualizations](#-visualizations)
- [🔮 Future Improvements](#-future-improvements)
- [👨‍💻 Author](#-author)

## 📊 Dataset

**Source**: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)

### Dataset Characteristics:
- **Training Set**: 891 passengers with survival labels
- **Test Set**: 418 passengers for predictions
- **Features**: 12 original features including passenger class, age, sex, fare, family relationships
- **Target**: Binary survival outcome (0 = Did not survive, 1 = Survived)

### Missing Data Challenges:
| Feature | Missing Count | Percentage | Strategy Applied |
|---------|---------------|------------|------------------|
| Age | 177 | 19.9% | Median imputation |
| Cabin | 687 | 77.1% | Feature engineering (Cabin_Known, Deck extraction) |
| Embarked | 2 | 0.2% | Mode imputation |

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.12+
pip install -r requirements.txt
```

### Required Libraries
```python
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
joblib>=1.1.0
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Ayushkr15/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

## 🏗️ Project Structure

```
Titanic-Survival-Prediction/
├── 📁 notebooks/
│   ├── 🔍 train_preprocess.ipynb    # Data cleaning & preprocessing
│   ├── 🧪 test_preprocess.ipynb     # Test data preprocessing
│   ├── 📈 graphs.ipynb              # Exploratory data analysis
│   ├── 🌲 rf_train_and_predict.ipynb # Random Forest implementation
│   └── 🧠 test_model.ipynb          # Model evaluation & comparison
├── 📁 model_training/               # Saved model artifacts (*.joblib)
├── 📁 visualizations/              # Generated plots and charts (*.png, *.jpg)
├── 📄 README.md                    # Project documentation
├── 📋 requirements.txt             # Python dependencies
└── 🚫 .gitignore                  # Git ignore patterns
```

## ⚙️ Data Preprocessing

### 🔧 Feature Engineering Pipeline

#### **Core Transformations:**
1. **Missing Value Treatment**
   - **Age**: Median imputation (28.0 years)
   - **Embarked**: Mode imputation ('S')
   - **Cabin**: Binary indicator + deck extraction

2. **Feature Creation** (37 Total Features)
   ```python
   # Family-based features
   FamilySize = SibSp + Parch + 1
   IsAlone = (FamilySize == 1)
   
   # Cabin information
   Cabin_Known = cabin.notna()
   Cabin_Deck = cabin.str[0]  # Extract deck level
   
   # Title extraction from names
   Title = name.str.extract('([A-Za-z]+)\.')
   ```

3. **Categorical Encoding**
   - One-hot encoding for Pclass, Embarked, Cabin_Deck
   - Binary encoding for Sex (male=0, female=1)

4. **Data Validation**
   - Feature alignment between train/test sets
   - Boolean to integer conversion for ML compatibility

## 🤖 Models & Performance

### 🏅 Model Comparison

| 🏆 Rank | Model | Accuracy | F1-Score | Precision | Recall | Key Strengths |
|---------|-------|----------|----------|-----------|--------|---------------|
| 🥇 | **Logistic Regression** | **81.6%** | **0.744** | 0.891 | 0.696 | Best overall performance, interpretable |
| 🥈 | Random Forest (Baseline) | 79.9% | 0.723 | 0.821 | 0.681 | Robust, handles non-linearity |
| 🥉 | Tuned Random Forest | 79.3% | 0.704 | 0.786 | 0.638 | Feature importance insights |

### 🔍 Model Details

#### **🏆 Champion: Logistic Regression**
- **Why it won**: Superior precision (89.1%) with balanced recall
- **Prediction pattern**: 256 non-survivors, 162 survivors
- **Key advantage**: Linear interpretability with excellent performance

#### **🌳 Random Forest Analysis**
- **Feature Importance Discovery**:
  - **Sex**: 33% (Most discriminative - "Women & Children First")
  - **Age**: 19% (Age-based survival patterns)  
  - **Fare**: 19% (Socioeconomic disparity)
  - **Pclass**: 8% (Class-based evacuation priority)

## 📈 Key Insights

### 🔍 Historical Pattern Discovery

1. **👫 Gender Disparity**: 74.2% female survival vs 18.9% male survival
   - Confirms "Women and Children First" evacuation protocol

2. **💰 Class Inequality**: Clear survival hierarchy
   - **1st Class**: 63.0% survival rate
   - **2nd Class**: 47.3% survival rate  
   - **3rd Class**: 24.2% survival rate

3. **👶 Age Factor**: Younger passengers (20-40) show higher survival density

4. **🤝 Model Agreement**: Strong consistency across algorithms validates approach

## 🚀 Usage

### 🔄 Training Pipeline
```python
# 1. Data Preprocessing
python train_preprocess.ipynb
python test_preprocess.ipynb

# 2. Model Training
python rf_train_and_predict.ipynb
python test_model.ipynb

# 3. Evaluation & Comparison
# Results automatically saved as .joblib and .csv files
```

### 📊 Prediction Generation
```python
import joblib
import pandas as pd

# Load best model
model = joblib.load('model_training/best_logistic_regression.joblib')

# Load preprocessed test data
test_data = pd.read_csv('test_preprocessed.csv')

# Generate predictions
predictions = model.predict(test_data)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('my_submission.csv', index=False)
```

## 📁 Files Description

### 📓 Notebooks
- **`train_preprocess.ipynb`**: Complete data cleaning and feature engineering for training data
- **`test_preprocess.ipynb`**: Preprocessing pipeline for test data with alignment checks
- **`graphs.ipynb`**: Comprehensive EDA with survival analysis visualizations
- **`rf_train_and_predict.ipynb`**: Random Forest implementation with hyperparameter tuning
- **`test_model.ipynb`**: Model evaluation framework with cross-validation and metrics

### 🗂️ Generated Outputs
- **Model Files**: `*.joblib` - Trained models for reproduction
- **Submissions**: `*_submission.csv` - Competition-ready prediction files
- **Visualizations**: `*.png`, `*.jpg` - Plots for analysis and reporting

## 🎨 Visualizations

### 📊 Key Charts Generated
1. **Survival Rate by Gender** - Demonstrates gender disparity (74.2% vs 18.9%)
2. **Survival Rate by Class** - Shows socioeconomic patterns (63% → 47% → 24%)
3. **Age Distribution** - Reveals age-based survival patterns
4. **Feature Importance** - Random Forest feature ranking analysis
5. **Confusion Matrices** - Model performance breakdown
6. **Model Agreement Matrix** - Cross-validation consistency check

## 🔮 Future Improvements

### 🚀 Potential Enhancements
1. **🤖 Advanced Ensembling**: Stacking, boosting, and voting classifiers
2. **🧠 Deep Learning**: Neural networks for complex pattern recognition
3. **🔍 Feature Selection**: Recursive elimination and LASSO regularization
4. **⚙️ Hyperparameter Optimization**: Bayesian optimization for better tuning
5. **📊 External Data**: Additional historical context and passenger information

### 🎯 Performance Goals
- **Target**: >85% accuracy through ensemble methods
- **Focus**: Improve recall for survivor identification (reduce false negatives)
- **Approach**: Combine predictions from multiple model families

## 🏅 Competition Results

### 📈 Kaggle Submission Performance
- **Best Single Model**: 81.6% accuracy (Logistic Regression)
- **Submission Files**: 3 different approaches available
- **Ranking Strategy**: Primary submission uses best_lr model

## 🛡️ Model Validation

### ✅ Robustness Checks
- **Cross-Model Agreement**: >95% prediction consistency
- **Historical Validation**: Predictions align with documented survival rates
- **Feature Stability**: Consistent importance rankings across models
- **Generalization**: Strong performance on stratified validation sets

## 👨‍💻 Author

**Ayush Kumar Sinha**
- 🔗 **GitHub**: [@Ayushkr15](https://github.com/Ayushkr15)
- 📧 **Email**: [Contact](mailto:your-email@domain.com)
- 💼 **LinkedIn**: [Profile](https://linkedin.com/in/your-profile)
- 🎯 **Application**: AI/ML Intern Position

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing the Titanic dataset and competition platform
- **Dr. ViKi** for the AI/ML internship opportunity and technical challenge
- **Open Source Community** for the excellent machine learning libraries

---

<div align="center">
  <h3>🌟 If you found this project helpful, please give it a star! 🌟</h3>
  
  **Made with ❤️ for machine learning and data science**
  
  *"In the face of tragedy, data reveals the patterns of human survival - where gender, age, and social class created invisible lines between life and death on that fateful April night."*
</div>
