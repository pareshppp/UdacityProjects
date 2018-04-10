## Machine Learning Algorithms - Intuitive Understanding

### Decision Trees
<li> Decision Trees search the data and find the feature that best spits the data (classifies the best). 
<li> The data is then split across that feature. 
<li> Then within the subsets of the data, the Decision Trees search for the next best feature that best classifies those subsets of data. 
<li> This process goes on until the data cannot be split any more or until a pre-specified limit.

### Naive Bayes
<li> Naive Bayes assigns a probability to each feature based on how well the feature classifies the data.
<li> The probabilities of all the features are combined to calculate the final probability.
<li> The final probability is then used to classify the data.

### Regression
#### Gradient Descent
<li> Gradient Descent is the process of taking multiple steps to go from High Error to Low Error.
<li> Each step is taken in such a way that the Log Loss error is minimized.
<li> The step at which error cannot be minimzed any longer is the solution.

#### Linear Regression
<li> Linear Regression uses Gradient Descent to find the Best-Fit line along the data points.
<li> Initially, a random line is taken as the Fit line.
<li> The error for that Fit line is calculated using the Least Square method.
<li> Then the line is shifted using the Gradient Descent method to reduce the error.
<li> This process is repeated until the step is reached beyond which the error cannot be reduced any more.
<li> The Fit line for this step is the Best-Fit line for that dataset.

#### Logistic Regression
<li> Logistic Regression uses Gradient Descent to find the Best-Fit line that divides (classifies) the data points.
<li> Initially, a random line is taken as the Fit line.
<li> The error is calculated by using the mis-classified data points.
<li> Each data point produces a penalty based on its distance from the Fit line.
<li> The mis-classified data points have much higher penalty than the correctly-classified data points.
<li> The penalties are then added together to get the total penalty fot that Fit line.
<li> Then the line is shifted using the Gradient Descent method to reduce the error/total penalty.
<li> This process is repeated until the step is reached beyond which the error/total penalty cannot be reduced any more.
<li> The Fit line for this step is the Best-Fit line that divides/classifies the dataset.

### Support Vector Machine
<li> Support Vector Machine is similar to Logistic Regression in that it uses Gradient Descent to find the Best-Fit line that correctly classifies the data.
<li> But a data set can have many lines that correctly classify the data.
<li> To choose between these lines, SVM finds the line that is farthest from all the classes in the dataset while still correctly classifying the data.
<li> SVM finds data points from each class that are closest to the Fit line.
<li> Then it uses Gradient Descent to find the line that is farthest from each of those points.
<li> This process is repeated until the Best-Fit line is found.
<li> This helps SVM to be more generalizable to unseen data.

### Neural Networks
<li> Instead of using the single Best-Fit line, Neural Network uses multiple lines to classify the data.
<li> Each of these smaller Logistic Regression lines are calculated by different nodes.
<li> Each nodes produces output of whether a point is correctly classified or not.
<li> The output of all nodes is added together (AND) to get the final output.
<li> This process is repeated for each data point.

### Kernel Method
<li> Kernel Method is used when a line is insufficient to divide/classify the data properly.
<li> Instead, iit uses a Curve/Plane to divide/classify the data.
<li> Kernel Method is generally used in conjuction with SVM to find the optimal plane/curve.

### K-Means Clustering
<li> K-Means Clustering is used where the number of clusters (k) is previously known.
<li> k random points are taken as the center of the clusters.
<li> Their distance to all the data points are calculated and the data points closest to each mean-point form the cluster.
<li> The mean-point are then moved to the center of their respective clusters.
<li> The distance to the surrounding data points are calculated again 
<li> Then the data points are re-assigned to clusters based on their distances to the mean-points.
<li> The mean-point are then moved to the center of their respective clusters again.
<li> This process is repeated until no more changes are possible.

### Hierarchical Clustering
<li> Hierarchical Clustering is used when the number of clusters is not known beforehand. But the minimum distance between any two clusters is known.
<li> Hierarchical Clustering starts with creating mini clusters out of the two data points that have the closest.
<li> Similarly, all the points are clustered with their closest neighbouring data point.
<li> Any left over data points are then joined to the mini-clusters closest to them.
<li> The min-clusters then join with their closest neighbouring min-cluster to form larger clusters.
<li> This joinng continues until the distance between the clusters reaches the pre-defined distance.


