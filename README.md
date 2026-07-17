# Stability of Machine Learning Predictive Features Under Limited Data

## 📖 Overview

Machine Learning models often rely on large, complete, and high-quality datasets to produce accurate predictions. However, in many real-world applications such as healthcare, finance, cybersecurity, and risk analysis, collecting complete datasets is difficult. Missing values and limited data can reduce the reliability of machine learning models and lead to unstable feature selection.

This project analyzes the stability of predictive features under limited data conditions by comparing two popular machine learning algorithms: **Random Forest** and **Logistic Regression**. The project measures feature selection consistency using the **Jaccard Similarity Index** and visualizes how frequently each feature is selected across multiple training iterations.

---

## 🎯 Objectives

- Evaluate the stability of predictive features when training data is limited.
- Compare Random Forest and Logistic Regression feature selection.
- Measure feature stability using the Jaccard Similarity Index.
- Analyze feature selection frequency across multiple iterations.
- Demonstrate the importance of stable feature selection in machine learning.

---

## 🚀 Features

- Synthetic dataset generation using Scikit-learn
- Data preprocessing using StandardScaler
- Random Forest Classifier
- Logistic Regression Classifier
- Feature Importance Analysis
- Jaccard Similarity Calculation
- Feature Selection Frequency Visualization
- Performance comparison between machine learning models

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📂 Project Structure

```
stability-of-ml-predictive-features/
│
├── feature_stability.py
├── README.md
├── requirements.txt
├── images/
│   └── feature_selection_frequency.png
├── presentation/
│   └── Stability_of_Machine_Learning_Project.pptx
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Ibrahimsha02/stability-of-ml-predictive-features.git
```

Move into the project folder:

```bash
cd stability-of-ml-predictive-features
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python feature_stability.py
```

---

## 📊 Methodology

1. Generate a synthetic classification dataset.
2. Randomly create multiple limited-data subsets.
3. Standardize the dataset.
4. Train Random Forest and Logistic Regression models.
5. Select the Top-5 important features.
6. Calculate feature stability using the Jaccard Similarity Index.
7. Compute average stability across multiple iterations.
8. Plot feature selection frequency.

---

## 📈 Output

The program displays:

- Average Feature Stability of Random Forest
- Average Feature Stability of Logistic Regression
- Feature Selection Frequency Graph

Example Output:

```
Average Stability (Random Forest): 0.546
Average Stability (Logistic Regression): 0.570
```

---

## 📷 Sample Output

### Feature Selection Frequency

> Add the graph screenshot inside the `images` folder and display it here.

```markdown
![Feature Selection Frequency](images/Screenshot%202026-07-16%20181128.png)
```

---

## 📚 Machine Learning Concepts Used

- Classification
- Feature Selection
- Feature Importance
- Random Forest
- Logistic Regression
- Standardization
- Jaccard Similarity
- Stability Analysis

---

## 🎯 Applications

This project can be applied in:

- Healthcare Prediction
- Disease Diagnosis
- Financial Risk Analysis
- Credit Scoring
- Fraud Detection
- Cybersecurity
- Medical Research
- Data Science Research

---

## 🔮 Future Enhancements

- Use real-world healthcare datasets.
- Compare additional ML algorithms such as XGBoost and SVM.
- Integrate Explainable AI (SHAP and LIME).
- Build a web interface using Flask or Django.
- Deploy the project on the cloud.

---

## 📖 References

- Breiman, L. (2001). Statistical Modeling: The Two Cultures.
- Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Why Should I Trust You? Explaining the Predictions of Any Classifier.
- Domingos, P. (2012). A Few Useful Things to Know About Machine Learning.

---

## 👨‍💻 Author

**Ibrahim Sha**

- GitHub: https://github.com/Ibrahimsha02
- LinkedIn: *(Add your LinkedIn profile here)*

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
