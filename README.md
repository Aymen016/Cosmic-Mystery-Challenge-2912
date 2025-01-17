<h1>Cosmic Mystery Challenge</h1>
Welcome to the Cosmic Mystery Challenge! In this data science adventure, you'll be transported to the year 2912 where your skills are urgently needed to solve a cosmic enigma. Are you ready to embark on this journey to rescue lost passengers from the Spaceship Titanic's collision with a spacetime anomaly?

<h1> Recommended Competition </h1>
We highly recommend participating in the Titanic - Machine Learning from Disaster competition to acquaint yourself with the basics of machine learning and Kaggle competitions.

<h1> Challenge Overview </h1>
The Spaceship Titanic, an interstellar passenger liner, recently embarked on its maiden voyage carrying nearly 13,000 passengers to three newly habitable exoplanets. However, tragedy struck as the vessel collided with a spacetime anomaly near Alpha Centauri, transporting almost half of its passengers to an alternate dimension.

Your mission is to predict which passengers were transported by the anomaly using records recovered from the spaceship's damaged computer system. By doing so, you'll aid rescue crews in retrieving the lost passengers and altering history.
the major steps taken in the data preprocessing and modeling:

<h2> 1.Handling Missing Values: </h2>

The missing values in the 'Age' column are filled with the mean age of passengers to ensure we don't lose data.
Similarly, missing values in the 'HomePlanet' column are filled with the mode (most frequent value) of the column.
For numerical features like 'VIP', 'RoomService', etc., missing values are filled with the mean of each respective feature.
For the 'Destination' column, missing values are filled with the mode of the column.

<h2> 2.Data Type Conversion: </h2>

The 'HomePlanet' column is converted to a categorical data type.
The 'CryoSleep' column is converted to boolean type for easier handling.

<h2> 3.Handling Erroneous Data: </h2>

Rows where passengers are in cryogenic sleep are identified.
Erroneous rows with non-zero values in amenity-related columns (like 'RoomService', 'FoodCourt', etc.) are identified and removed.

<h2> 4.Extracting Information from 'Cabin' Column: </h2>

The 'Cabin' column is filled with 'Unknown' for missing values.
Relevant information like 'CabinLevel', 'CabinSection', and 'Cabinn' is extracted from the 'Cabin' column and stored in separate columns.
Missing values in these new columns are filled with the mode of each respective column.

<h2> 5.Exploratory Data Analysis (EDA): </h2>

Visualizations are created to explore the distribution of features like 'Age' and the distribution of transported vs. non-transported passengers.

<h2> 6.Model Training and Prediction: </h2> 

A RandomForestClassifier model is trained using features like 'Age', 'CryoSleep', 'ShoppingMall', and 'FoodCourt'.
The trained model is used to make predictions on the test dataset.
The predictions are saved to a CSV file for submission.
