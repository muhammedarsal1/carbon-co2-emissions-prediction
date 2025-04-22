# 🌍 Carbon CO₂ Emissions Prediction

This project analyzes and predicts carbon dioxide (CO₂) emissions data by country using machine learning techniques. The dataset is sourced from Kaggle and includes CO₂ emissions data across multiple countries and years.

## 📊 Dataset

- **Source**: [Kaggle - Carbon (CO₂) Emissions by Country](https://www.kaggle.com/datasets/ravindrasinghrana/carbon-co2-emissions)
- **Columns**:
  - `Region`: The continent or zone the country belongs to
  - `Country`: The specific country name
  - `Date`: Year of the emission record
  - `Metric Tons Per Capita`: Emissions per person
  - `Kilotons of CO₂`: Total national emissions

## 🎯 Project Goal

To:
- Explore the dataset (EDA)
- Visualize emission trends
- Train a machine learning model to predict CO₂ emissions (`Kilotons of CO₂`)

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Google Colab

## 🔍 Exploratory Data Analysis (EDA)

- Checked null values and duplicates
- Visualized distribution using histograms and KDE plots
- Encoded categorical features using `LabelEncoder`
- Converted `Date` to datetime format
- Displayed correlation matrix using heatmap

## 🤖 Model Training

- Used **Linear Regression** to predict CO₂ emissions
- Feature scaling with `StandardScaler`
- Trained and tested using `train_test_split`
- Evaluated model with **R² score**

## 📈 Results

- The trained model can predict CO₂ emissions based on yearly trends
- Accuracy measured using R² score for training data

## 🧪 Sample Code (Model Training)

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
