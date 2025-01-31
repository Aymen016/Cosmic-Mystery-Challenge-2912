# 🚀 **Spaceship Titanic: Predicting Passenger Transported Status**

![Untitled design (1)](https://github.com/user-attachments/assets/5a12dbd4-5e1b-4df2-9011-17d1dc11c37b)


## 📚 **Project Overview**
This project is based on the **Spaceship Titanic** dataset, which contains passenger information from a spaceship journey. The goal is to predict whether a passenger was **transported** to a different dimension, based on various features such as home planet, age, cryogenic sleep status, and amenities usage.

We preprocess the dataset, handle missing values, explore correlations, and visualize the distribution of features to build and train a model for **classification**. 

## 🧑‍💻 **Installation and Setup**

### **Prerequisites**
Ensure you have **Python** and the required libraries installed. You can install the necessary dependencies by running:

```bash
pip install numpy pandas seaborn matplotlib
```

### 📚 **Dataset**
This project uses the **Spaceship Titanic** dataset, available on Kaggle. It contains two main files:
- **train.csv**: The training dataset containing labeled data.
- **test.csv**: The testing dataset for prediction (no labels).

### **Loading the Data**
The data is loaded using **Pandas** as follows:

```python
train_data = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
```

## 🧹 **Data Preprocessing**
Data preprocessing is a crucial step before feeding the data into a machine learning model. Below are the steps involved:

### **Handling Missing Values**:
- For numerical features such as **Age**, missing values are filled with the mean.
- For categorical features like **HomePlanet**, missing values are filled with the mode (most frequent value).
- **Amenities** columns (RoomService, FoodCourt, Spa, VRDeck) with missing values are filled with the mean of each respective feature.

### **Categorical Encoding**:
- **HomePlanet** and **CryoSleep** are converted into **category** types, and **CryoSleep** is also converted to boolean values.

### **Feature Engineering**:
- We create new features like **CabinLevel**, **CabinSection**, and **Cabinn** by splitting the **Cabin** feature.

### **Error Handling**:
- We identify rows where passengers in cryogenic sleep have non-zero values in amenities-related columns and handle these erroneous rows by removing them.

## 📊 **Exploratory Data Analysis (EDA)**
Visualization helps us understand the underlying patterns in the data. Below are some key visualizations:

### **Distribution of Passengers by HomePlanet**:
- This bar chart shows how passengers are distributed across different home planets.

- ![image](https://github.com/user-attachments/assets/660d1492-39d1-47dd-8bd1-a326236aa335)


### **Distribution of Passengers by Destination**:
- Visualizes the number of passengers going to each destination.

- ![image](https://github.com/user-attachments/assets/af11beb3-a8f8-4c08-a529-e0513da9a05c)


### **Age Distribution by Transported Status**:
- A boxplot showing the distribution of **Age** for passengers who were transported vs. those who were not.

- ![image](https://github.com/user-attachments/assets/fc051f93-75d3-4ef2-87f0-1bea95dcd376)


### **Correlation Heatmap**:
- Displays the correlation between numerical features in the dataset to identify potential relationships.

- ![Screenshot 2025-01-31 170926](https://github.com/user-attachments/assets/6dff3f47-c0ec-492f-9299-a60aae937103)


### **Transportation Status Distribution**:
- A countplot showing the number of passengers who were and weren't transported.

- ![image](https://github.com/user-attachments/assets/1e3f7359-0c63-4e7f-8617-2f174d9cd034)


# 🔍 **Feature Engineering & Transformation**
- **Age Transformation**: Missing values in **Age** are filled with the mean age.
- **Cabin Transformation**: Splitting the **Cabin** column into **CabinLevel**, **CabinSection**, and **Cabinn** for further analysis.
- **CryoSleep Transformation**: Convert the **CryoSleep** column to boolean values and handle erroneous data where passengers in cryo sleep have non-zero values in amenities columns.

## ⚙️ **Modeling**
The ultimate goal is to predict the **Transported** status using various features. Below is an example of how you might implement a machine learning model:

```python
# Example: Model training with RandomForestClassifier or any other classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data for model
X = train_data.drop(columns=['Transported'])
y = train_data['Transported']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score}")
```

## 📈 **Model Evaluation**
We use **accuracy** and other classification metrics to evaluate the model. We ensure the model generalizes well by testing on the **test dataset**.

## 🚧 **Future Improvements**
- **Hyperparameter Tuning**: Experiment with different algorithms and hyperparameter optimization techniques such as **GridSearchCV** or **RandomizedSearchCV**.
- **Feature Selection**: Use techniques like **Recursive Feature Elimination (RFE)** or **PCA** to reduce the feature space and improve model performance.
- **Cross-validation**: Implement **k-fold cross-validation** for better performance evaluation.

## 💬 **Conclusion**
Through this project, we successfully implemented data preprocessing and visualization techniques to clean and analyze the **Spaceship Titanic** dataset. The next steps involve building predictive models and comparing their performance to identify the best algorithm for the prediction of **Transported** status. We also plan to continue refining the model by testing more complex algorithms and adding feature engineering techniques.

## 📥 **License**
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Aymen016/Cosmic-Mystery-Challenge-2912/blob/master/LICENSE) file for details.

