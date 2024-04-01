# Machine-learning

Q1. Data Preprocessing
1. Assign a type to each of the following features (a) Model, (b) Type, (c) Max. Price and (d) Airbags from
the following: ordinal/nominal/ratio/interval scale.
-> I have categorized the model and Type into the nominal scale because there is no meaningful
order in model and Type. and for “Max.Price” I have assigned the interval scale because this feature
can be ranked based on Price and also it is in some interval. And for “Airbags” I have assigned the
Ratio scale. It can be categorized.
2. Write a function to handle the missing values in the dataset (e.g., any NA, NaN values).
-> In the given data only two features have “nan” value which is “Rear.seat.room” and
“Luggage.Room”.
- here “Rear.seat.room” have only two “nan” value so I have removed this data row from the
Dataframe.
- And for the “Luggage.Room” there is 9 “nan” value. So, I have replaced this data by mean of that
column.
3. Write a function to reduce noise (any error in the feature) in individual attributes.
-> In the given data for reducing the noise we have to firstly identify the numerical columns.
- After identifying numerical columns we have to identify outliers for all the particular columns. As I
wrote the code for identifying outliers and replaced the outliers by the median.
4. Write a function to encode all the categorical features in the dataset according to the type of variable
jointly.
-> For encoding columns, first need to identify the categorical feature. After that I need to identify how
many unique values there are in that feature.
- Here I removed the “Model” feature because it has 91 unique values. It's almost similar to the row in
the dataframe so no need to encode this feature.
- And here I used the OneHotEncoder to encode these categorical features. Because, in
OneHotEncoder, encoding happens in the form of 0 and 1. By combining all the features(jointly).and
convert categorical features into binary data and we are not using the LableEncoder because it is good
for individual feature not for multiple.
5. Write a function to normalize / scale the features either individually or jointly.
-> For scaling/normalize the feature I have used the “StandardScaler” because it contains the outliers
but we have replaced outliers by its median.
- Here I did not use MinMaxScaler because it’s helpful when outliers in the feature.
6. Write a function to create a random split of the data into train, validation and test sets in the ratio of
[70:20:10].
-> Here I have used the “train_test_split” from the model_selection which is part of the scikit learn
library.- First I split the data into the train and test into 90:10 ratio after that again I split the train data
into 2 part one is train and other is validation 78:22 ratio. In final result is
train:validation:test=[70:20:10].



Q2a: Linear Regression Task.
1. Implement linear regression using the inbuilt function “LinearRegression” model in sklearn.
-> Load the dataset from excel file using pandas dataframe and reshape the feature in 2D to use in the
LinearRegression.
2. Print the coefficient obtained from linear regression
and plot a straight line on the scatter plot.
-> After applying the model on test data i got the
Coefficient = 61.27218654
Intercept = -39.06195591884392
R-Squared error = 0.9891
3. Now, implement linear regression without the use of
any inbuilt function (write it from scratch).
-> I have used the least-squares method here to
calculate the coefficient and intercept. After that I also
find the R-squared error that will determine the
accuracy of the model.
4. Compare the results of 1 and 3 graphically.
-> I got the same regression line in both the linear regression.

Q2b: Logistic Regression Task.
1. Split the dataset into training set and test set in the ratio of 70:30 or 80:20
-> Here I have used the Purchased column as a targeted variable and after that split the data.
2. Train the logistic regression classifier (using inbuilt function: LogisticRegression from sklearn).
-> Here 'newton-cholesky’ and ‘liblinear’ solver in logistic regression is for small dataset.
3. Print the confusion matrix and accuracy.
-> Here I got 88.75% accuracy in the newton-cholesky, and 73.75% accuracy in ‘liblinear’ solver. But
here newton-cholesky is used for non-linear relationships. And liblinear is used for non-linear
relationships. So, here it is better to use newton-cholesky.



Q3: SVM
1. Store the dataset in your google drive and in Colab file load the dataset from your drive.
2. Check the shape and head of the dataset.
3. Age, Experience, Income, CCAvg, Mortgage, Securities are the features and Credit Card is your
Target Variable.
a. Take any 3 features from the six features given above.
b. Store features and targets into a separate variable.
c. Look for missing values in the data, if any, and address them accordingly.
d. Plot a 3D scatter plot using Matplotlib.
-> Here I have used the Experience, Age and Income feature. and I couldn't found any
missing value or null value.
4. Split the dataset into 80:20. (3 features and 1 target variable).
5. Train the model using scikit learn SVM API (LinearSVC) by setting the regularization parameter C as C
= {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}.
a. For each value of C, print the score on test data.
b. Make the prediction on test data.
c. Print the confusion matrix and classification report.
-> Here I found that when regularization parameter is low then accuracy is high and when the
regularization parameter is high then accuracy is low.
6. Use gridSearchCV - a cross-validation technique to find the best regularization parameters (i.e.: the
best value of C).
In the report provide your findings for the output generated for all the kernels used and also describe
the changes that happened after changing the regularization hyperparameter.
-> For gridSearchCV here I have used a linear kernel and also used gamma and regularization
parameters. So, after getting output we get the best value of C = 0.0001.
- Changing in regularization parameter affects the accuracy as when value goes up then the street of
the support vectors are going narrow(It causes the overfitting) and the value of the regularization
parameter if reducing the street of the support vectors are going wider. So, it causes underfitting.
- And from the confusion matrix we have found that we get around 30% False negative predictions and
around 70% True positive predictions.



Q4: Decision Tree and Random Forest.
1. Visualize the distribution of each feature and the class distribution.
-> Load the IRIS dataset from the sklearn and load into dataframe using pandas and
also separate the targeted variable which is “species”
-Here I have used the seaborn and matplotlib library to plot the feature distribution and class
Distribution
2. Encode the categorical target variable (species) into numerical values.
-> Here i’ve used the LabelEncoder to encode the target variable(species)
3. Split the dataset into training and testing sets (use an appropriate ratio).
-> Here I have used an 80:20 ratio to split the IRIS dataset.
4. Decision Tree Model
i. Build a decision tree classifier using the training set.
-> To build a decision tree use DecisionTreeClassifier from sklearn and tune the hyperparameter to
achieve the high accuracy here. I have used the max_depth=3, max_leaf_nodes=4 this parameter.
For this parameter I achieve the highest accuracy.
ii. Visualize the resulting decision tree.
-> To draw a decision tree I have used the plot_tree from the sklearn.
iii. Make predictions on the testing set and evaluate the model's performance using appropriate
metrics (e.g., accuracy, confusion matrix).
-> For the above parameters I have achieved 100% accuracy, and also print the confusion matrix and
classification report. Which is good



6. Random Forest Model
i. Build a random forest classifier using the training set.
ii. Tune the hyperparameters (e.g., number of trees, maximum depth) if necessary.
iii. Make predictions on the testing set and evaluate the model's performance using appropriate metrics
and compare it with the decision tree model.
-> For a random forest classifier I tried for different parameters max_depth, max_leaf_nodes,
n_estimator but i can’t see any changes in accuracy. So, I removed it.
- I get the same accuracy in both decision tree and random forest classifier.
