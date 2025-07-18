# Heart Disease Prediction using Logistic Regression

This project aims to predict the likelihood of heart failure using a logistic regression machine learning model trained on real-world clinical data.

## 📁 Project Files

- `Heart_Failure_Prediction_ML_Project.ipynb` – Jupyter Notebook containing the complete code for data preprocessing, model training, evaluation, and predictions.
- `heart.csv` – Dataset used in the project.
- `logistic_heart_model.pkl` – Trained and serialized logistic regression model for deployment and reuse.

## 📊 Model Evaluation

- **Accuracy**: `89.8%`
- **Confusion Matrix**:
- **Classification Report**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.89      | 0.94   | 0.91     | 68      |
| 1     | 0.91      | 0.84   | 0.88     | 50      |
| **Overall** | **-** | **-** | **0.90** | **118** |

## 📌 Steps Included

1. Importing libraries and loading dataset
2. Exploratory data analysis (EDA)
3. Data preprocessing and feature selection
4. Splitting data into train and test sets
5. Model training using Logistic Regression
6. Model evaluation (confusion matrix, accuracy, precision, recall)
7. Model saving with pickle

## 🛠️ Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- scikit-learn
- Pickle

## 💡 Future Improvements

- Add hyperparameter tuning
- Try other models like Random Forest, XGBoost, or SVM
- Integrate with a web interface (Flask or Streamlit)

## 📥 How to Use

```bash
# Clone the repository
git clone https://github.com/baqar360/Heart-Disease-Prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Heart_Failure_Prediction_ML_Project.ipynb
