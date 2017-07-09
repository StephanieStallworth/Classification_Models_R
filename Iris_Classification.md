# Iris Machine Learning
Stephanie Stallworth  
April 20, 2017  



###**Step 1: PreProcessing**

**1.1 Load `iris` dataset**

The `iris` data was used for this analysis. This dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.


```r
# Attach iris data set to environment
data(iris)

#Rename data set
dataset<-iris
```


**1.2 Load Caret Package**

The caret package in R was utlizied to build the models.  This package provides a consistent interface into hundreds of machine learning algorithms and provides useful convenience methods for data visualization, data resampling, model tuning and model comparison, among other features. It's a must have tool for machine learning projects in R.


```r
# Load package
library(caret)
```


###**Step 2: Create Validation Set**

I split the loaded `iris` dataset into two parts:   
-  80% used to train the models   
-  20% held back as a validation dataset   

Creating a validation was critical in determining whether the built models were any good.

After using statistical methods to ***estimate*** the accuracy of the models created on unseen data, I would need a more concrete accuracy estimate of the best model on unseen data by evaluating it on ***actual*** unseen data in the validation set. That is, I held back some data that the algorithms did not see (the validation set) and used that data to get a second and independent idea of how accurate the best model would actually be.



```r
# Create a list of 80% of the rows in the original dataset we can use for training
validation_index<-createDataPartition(dataset$Species, p =0.80, list = FALSE)

# Select 20% of the data for validation
validation<-dataset[-validation_index, ]

# Use the remaining 80% of data to train and test the models
dataset<-dataset[validation_index, ]
```

###**Step 3: Summarize Dataset**

I then looked at the data in a number of different ways:  

1. Dimensions of the dataset  
2. Types of the attributes  
3. Peek at the data itself  
4. Levels of the class attribute  
5. Breakdown of the instances in each class  
6. Statistical summary of all attributes  

**3.1 Dimensions of the Dataset**

```r
# Dataset dimensions
dim(dataset)
```

```
[1] 120   5
```
**3.2 Types of Attributes**  
The types of attributes that reside in the data were identified next.  Knowing the types is important as it will give an idea of how to better summarize the data and what transformations I might need to apply to prepare the data before it is modeled.

```r
# List types for each attribute
sapply(dataset,class)
```

```
Sepal.Length  Sepal.Width Petal.Length  Petal.Width      Species 
   "numeric"    "numeric"    "numeric"    "numeric"     "factor" 
```
**3.3 Peek at the data**  
For good measure, I took a quick look at the first seven rows to get a better feel for what the data looks like.

```r
# View first five rows of the data
head(dataset)
```

```
  Sepal.Length Sepal.Width Petal.Length Petal.Width Species
1          5.1         3.5          1.4         0.2  setosa
2          4.9         3.0          1.4         0.2  setosa
3          4.7         3.2          1.3         0.2  setosa
4          4.6         3.1          1.5         0.2  setosa
5          5.0         3.6          1.4         0.2  setosa
6          5.4         3.9          1.7         0.4  setosa
```

**3.4  Levels of the Factor Variables**  
`Species` was the only factor variable in the dataset, so I investigated further to identify its levels. `Species` had three levels so this is a multi-class or a multinomial classification problem.  If there were only two levels, it would have been a binary classification problem.  


```r
levels(dataset$Species)
```

```
[1] "setosa"     "versicolor" "virginica" 
```

**3.5 Class Distribution**  
I then determined the number of instances that belong to each class of `Species` as an absolute count and as a percentage. Note that each class had the same number of instances (40 or 33% of the dataset).

```r
# Summarize the class distribution
percentage<-prop.table(table(dataset$Species)) * 100
cbind(freq = table(dataset$Species), percentage = percentage)
```

```
           freq percentage
setosa       40   33.33333
versicolor   40   33.33333
virginica    40   33.33333
```

**3.6 Statistical Summary**  


```r
# Summarize attribute distributions
summary(dataset)
```

```
  Sepal.Length   Sepal.Width     Petal.Length    Petal.Width   
 Min.   :4.40   Min.   :2.000   Min.   :1.200   Min.   :0.100  
 1st Qu.:5.10   1st Qu.:2.800   1st Qu.:1.575   1st Qu.:0.300  
 Median :5.80   Median :3.000   Median :4.250   Median :1.300  
 Mean   :5.83   Mean   :3.052   Mean   :3.725   Mean   :1.192  
 3rd Qu.:6.40   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
 Max.   :7.70   Max.   :4.400   Max.   :6.600   Max.   :2.500  
       Species  
 setosa    :40  
 versicolor:40  
 virginica :40  
                
                
                
```

###**Step 4: Visualize Dataset**

After getting a basic feel for the data, I extended that understanding with some visualizations:    
1. Univariate plots to better understand each attribute  
2. Multivariate plots to better understand the relationships between attributes  

**4.1 Univariate Plots**

I started with univariate plots(i.e. plots of each individual variable).

With visualization, it is helpful to have a way to refer to just the input attributes and just the output attributes. So I set up variables calling the input attributes x and the output attribute (or `Species` class) y.


```r
# Split input and output
x<-dataset[,1:4]
y<-dataset[ ,5]
```

Given that the input variables were numeric, I created box-and-whisker plots to view the distribution of each attribute.



```r
# Boxplot for each attribute on one image
par(mfrow=c(1,4))
  for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
  }
```

![](Iris_Machine_Learning_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

I then created a bar plot of the output variable (`Species`) to get a graphical representation of the class distribution.  This was  uninteresting as they were evenly distributed across the classes, confirming what was learned from the frequency table created earlier in the analysis.  

```r
# Barplot of class breakdown
library(dplyr)
library(ggplot2)

dataset %>% ggplot(aes(x= y)) + geom_bar() +labs(x = "Iris Flower Species")
```

![](Iris_Machine_Learning_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

**4.2 Multivariate Plots**

After plotting each individual attribute, I explored the interaction *between* the variables by looking at scatter plots of all attributes pairs with points colored by class. Because the scatter plots show that points for each class are generally separate,  ellipses were added around them to highlight this clear relationship between the input attributes (trends) and between attributes and the class values (ellipses).


```r
# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")
```

![](Iris_Machine_Learning_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

I then re-visited the box-and-whisker plots for each input variable, but this time breaking it out into separate plots for each `Species` class. This faceting helped tease out obvious linear separations between the classes and revealed that there are clearly different distributions of the attributes for each `Species` class.  


```r
# Box and whisker plots for each attribute
featurePlot(x = x, y = y, plot = "box")
```

![](Iris_Machine_Learning_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

The distribution of each attribute was explored further with probability density plots. Again, like the box and whisker plots above, the density plots were broken down by `Species` class,  Sometimes histograms are good for this, but I chose probability density plots in this case to give nice smooth lines for each distribution. Like the box plots, the difference in distribution of each attribute by class is apparent. It was also worth noting the Gaussian-like distribution (bell curve) of each attribute.
 

```r
# Density plots for each attribute by species class value
scales<-list(x = list(relation = "free"), y = list(relation = "free"))
featurePlot(x = x, y = y, plot = "density", scales = scales)
```

![](Iris_Machine_Learning_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

###**Step 5: Algorithm Evaluation**
Next, I created models of the data and estimated their accuracy on unseen data.
 
This was a three step process:  

1. Set-up the test harness to use 10-fold cross validation  
2. Build 5 different models to predict species from flower measurements  
3. Select the best model  

**5.1 Test Harness**  

I used 10-fold cross validation to estimate accuracy. This split the dataset into 10 parts (train in 9 and test on 1) and then released for all combinations of train-test splits.  The process was repeated 3 times for each of the 5 algorithms, with different splits of the data into 10 groups to get more accurate estimates.

As mentioned, the "Accuracy" metric was used to evaluate the models. This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate).


```r
# Run algorithms using 10-fold cross validation
control<-trainControl(method = "cv", number = 10)
metric<-"Accuracy"
```

**5.2 Build Models**  

It was initially unknown which algorithms would work well on this problem or what configurations to use. The plots suggested that some of the classes are partially linearly separable in some dimensions, so generally good results were expected.  

Five different algorithms were evaluated:

1. Linear Discriminant Analysis (LDA)  
2. Classification and Regression Trees (CART)  
3. k-Nearest Neighbors (kNN)  
4. Support Vector Machines (SVM) with a linear kernel  
5. Random Forest (RF)  

This was a good mixture of simple linear (LDA), nonlinear (CART, kNN) and complex nonlinear methods (SVM, RF). The random number seed was reset before each run to ensure evaluation of each algorithm was performed using exactly the same data splits and results were directly comparable.

The five models were built and saved as variables in the work space.

**Linear Algorithms: LDA**

```r
# Linear Discriminant Analysis (LDA)  
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
```

**Nonlinear Algorithms: CART and kNN**

```r
# Classification and Regression Trees (CART)
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

# k-Nearest Neighbors (kNN)
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
```

**Advanced Algorithms: SVM and RF**

```r
# Support Vector Machines (SVM)
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)


# Random Forest (RF)
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)
```

**5.3 Select Best Model**

Once the five models and accuracy estimations for each were created, my next task was to compare the models and select the most accurate.

To do this, I created a list of the fitted models and passed these results to the summary function to get an output displaying the accuracy of each classifier along with other metrics, like Kappa.


```r
# Summarize model accuracy for each model
results <- resamples(list(lda=fit.lda,cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
```

```

Call:
summary.resamples(object = results)

Models: lda, cart, knn, svm, rf 
Number of resamples: 10 

Accuracy 
       Min. 1st Qu. Median   Mean 3rd Qu. Max. NA's
lda  0.9167  0.9375 1.0000 0.9750  1.0000    1    0
cart 0.8333  0.9167 0.9167 0.9333  0.9792    1    0
knn  0.9167  0.9375 1.0000 0.9750  1.0000    1    0
svm  0.9167  0.9167 0.9583 0.9583  1.0000    1    0
rf   0.9167  0.9167 0.9583 0.9583  1.0000    1    0

Kappa 
      Min. 1st Qu. Median   Mean 3rd Qu. Max. NA's
lda  0.875  0.9062 1.0000 0.9625  1.0000    1    0
cart 0.750  0.8750 0.8750 0.9000  0.9688    1    0
knn  0.875  0.9062 1.0000 0.9625  1.0000    1    0
svm  0.875  0.8750 0.9375 0.9375  1.0000    1    0
rf   0.875  0.8750 0.9375 0.9375  1.0000    1    0
```
I then created a plot of the model evaluation results and compared the spread as well as the mean accuracy of each model. It is important to note that there is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation), which is why the *mean* accuracy estimates had to be compared.  The most accurate model in this case was LDA, given it had the highest mean accuracy with the smallest spread.


```r
# Compare accuracy of models
dotplot(results)
```

![](Iris_Machine_Learning_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

Since LDA was identified as the best model, results for just LDA were summarized. The output gave a nice summary of what was used to train the model and the mean and standard deviation (SD) accuracy achieved, specifically 97.5% accuracy +/- 4%


```r
# Summarize Best Model
print(fit.lda)
```

```
Linear Discriminant Analysis 

120 samples
  4 predictor
  3 classes: 'setosa', 'versicolor', 'virginica' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 108, 108, 108, 108, 108, 108, ... 
Resampling results:

  Accuracy  Kappa 
  0.975     0.9625

 
```

###**Step 6: Make Predictions**

LDA was the most accurate model on the training set, but I had to determine the model's accuracy on the validation set to get an independent final check on the accuracy of the best model. As best practice, I kept a validation set just in case of overfitting to the training set or a data leak, as both would have resulted in an overly optimistic result.

**The LDA model was ran directly on the validation set and results were summarized in a confusion matrix. The accuracy was 100%. It was a small validation dataset (20%), but this result was within the expected margin of 97% +/-4% suggesting LDA was an accurate and a reliable model.**


```r
# Estimate skill of LDA on the validation dataset
predictions<-predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)
```

```
Confusion Matrix and Statistics

            Reference
Prediction   setosa versicolor virginica
  setosa         10          0         0
  versicolor      0          9         0
  virginica       0          1        10

Overall Statistics
                                          
               Accuracy : 0.9667          
                 95% CI : (0.8278, 0.9992)
    No Information Rate : 0.3333          
    P-Value [Acc > NIR] : 2.963e-13       
                                          
                  Kappa : 0.95            
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: setosa Class: versicolor Class: virginica
Sensitivity                 1.0000            0.9000           1.0000
Specificity                 1.0000            1.0000           0.9500
Pos Pred Value              1.0000            1.0000           0.9091
Neg Pred Value              1.0000            0.9524           1.0000
Prevalence                  0.3333            0.3333           0.3333
Detection Rate              0.3333            0.3000           0.3333
Detection Prevalence        0.3333            0.3000           0.3667
Balanced Accuracy           1.0000            0.9500           0.9750
```






