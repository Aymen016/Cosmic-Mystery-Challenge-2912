# 🚀 **Spaceship Titanic: Predicting Passenger Transported Status**

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

### **Distribution of Passengers by Destination**:
- Visualizes the number of passengers going to each destination.

### **Age Distribution by Transported Status**:
- A boxplot showing the distribution of **Age** for passengers who were transported vs. those who were not.

### **Correlation Heatmap**:
- Displays the correlation between numerical features in the dataset to identify potential relationships.

### **Transportation Status Distribution**:
- A countplot showing the number of passengers who were and weren't transported.

## 📈 **Visualizing the Data**
Here are some visualizations used for understanding the dataset:

```python
plt.figure(figsize=(12, 8))
sns.countplot(x='HomePlanet', data=train_data)
plt.title('Distribution of Passengers by HomePlanet')
plt.xlabel('HomePlanet')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

![image](https://github.com/user-attachments/assets/660d1492-39d1-47dd-8bd1-a326236aa335)


```python
plt.figure(figsize=(12, 8))
sns.countplot(x='Destination', data=train_data)
plt.title('Distribution of Passengers by Destination')
plt.xlabel('Destination')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
```

![image](https://github.com/user-attachments/assets/af11beb3-a8f8-4c08-a529-e0513da9a05c)
