Q1. Download the following dataset diabetes (1).csv - Google Drive.
-> Data preprocessing: In the dataframe pragnancies column does not matter and Insulin is not
taken because there is lot of missing data in insulin so it's good to drop so, dropping above two
columns
- And after that I couldn’t find any null or zero value in the dataframe and also checked for
correlation of the columns by using heatmap but couldn’t find any correlation.
1. Find the optimum number of principal components for the features in the
above-mentioned data.
-> Here first apply the standardscaler to normalize all the column to get into same scale
Because here all the features are in some range so it’s better to normalize
- After that, I apply the PCA to find the principal number of components and I get to
know that we get 0.80 to 0.90 variance when we select the 3 components.
2. Use any two regression models of your choice and find the prediction accuracy and
error between the reduced data (with optimum number of principal components) and the
complete data.
-> Below is a snapshot of the R2 score and mean squared error for linear regression and
Random forest.
- Here as you can see there is very little difference of R2-score and Mean squared error
in linear and random forest regression when we do not use the principal components.
- But when we use the principal component then we get the higher R2-score as
compared to the random forest



Q2. We will use the fashion-MNIST dataset for this question (you can download it from any
other source also including libraries). Flatten and preprocess the data (if required) before
starting the tasks. It will become a 784 dimensional data with 10 classes, more details are
available in the link.
1. Train the k-means model on f-MNIST data with k = 10 and 10 random 784 dimensional
points (in input range) as initializations. Report the number of points in each cluster.
-> Here I have used the random initializer to initialize the data point for each cluster.
After that, I count the predicted label of the data by using the bincount function from
numpy.
2. Visualize the cluster centers of each cluster as 2-d images of all clusters.
3. Visualize 10 images corresponding to each cluster.
-> Here I can see in the plot that all the trousers, shirt, t-shirt, shoes are clustering well
except women’s heels and handbags.
4. Train another k-means model with 10 images from each class as initializations , report
the number of points in each cluster and visualize the cluster centers.
-> Here first find the 10 data points from each cluster and after that find the mean of
those 10 data points and apply that centroid as initialization of each cluster.
5. Visualize 10 images corresponding to each cluster.
-> Here I can see in the plot that all the trousers, shirt, t-shirt, shoes are not clustering
well as compared to when we have used the 1 data point from each cluster.
6. Evaluate Clusters of part a and part d with Sum of Squared Error (SSE) method. Report
the scores and comment on which case is a better clustering.
-> I get good accuracy(accuracy is good when the sum of squared error is low) when I
select the 1 data point from each cluster as compared to selecting the centroid of the 10
data points from each cluster.



Q3. You have been given a dataset here. It consists of different characteristics of dry beans
(consider only: area, perimeter, axes lengths, eccentricity, roundness, aspect ratio, and convex
area - 7 features). You need to perform classification into different varieties (Cali, Bombay,
Barbunya, etc.). For this classification, you need to use a multi-layer perceptron.
1. Preprocess & visualize the data. Create train, val, and test splits but take into
consideration the class distribution (Hint: Look up stratified splits).
-> here we can see that area is highly correlated with Perimeter, MajorAxisLength,
ConvexArea and MinorAxisLength and aspectration is also highly correlated with the
Eccentricity so it’s better to remove this columns.
- here 91 outliers and also wants to scale these features(because here all features are
not in the same scale. if we don't scale the feature then weights are going to fluctuate
hence it affects the gradient descent. so scaling helps gradient descent smoothly
converge to minima )
- so, removing outliers or replacing by median is not sufficient way to get efficient output
so here I am using robust scaler that scale the feature by using median
- And also encode the label feature (that helps neural network to understand different
classes)
- Here I use the stratify attribute in the train_test_split function to maintain the
consistency in the data splitting process.
2. Now experiment with different activation functions (at least 3) and comment (in the
report) on how the accuracy varies. Create plots to support your arguments.
-> Here I used relu, sigmoid and tanh as activation functions for all layers but I find that if
we use the 1 hidden layer then sigmoid activation function accuracy is good as
compared to the relu and tanh.
3. Experiment with different weight initialization: Random, Zero & Constant. Create plots to
support your arguments.
-> here I get very low accuracy when I initialize the weights with zeros because if all the
weights are zero then no matter how many nodes output at the second layer might
slightly change, this will also happen when we initialize the weights with ones. in both
this case zeros and ones neural network not efficiently find the pattern in the data
- But I get very good accuracy when I initialize the weights with random numbers so, in
this scenario output at the all nodes of all hidden layers is different so the neural network
finds patterns in the data.
4. Change the number of hidden nodes and comment upon the training and accuracy.
Create plots to support your arguments.
-> Here I choose different number of nodes in the hidden layer but as i see in the plot
that if network has few nodes than accuracy of the model is less means underfitting and
while increasing number of nodes in the hidden layer then accuracy is increasing but
after certain point accuracy is saturated means when number of nodes increasing then
there may be chance that model is overfitted. So, here best accuracy we get at the 32 to
64 number of nodes in the hidden layer.
