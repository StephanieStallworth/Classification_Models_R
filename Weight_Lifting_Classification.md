# Weight Lifting Tree Based Classification
Stephanie Stallworth  
April 8, 2017  



### **Executive Summary**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. The frequency an activity is performed is often quantified, but how well that activity is performed is rarely examined.

This analysis uses data from accelerometers attached to 6 participants who were asked to perform barbell lifts correctly and incorrectly 5 different ways. The "classe" variable corresponds to how the exercise was performed by the participants with"A" denoting correct execution and the other 4 classes (B,C,D,and E) corresponding to common mistakes.

My objective is to build a model to predict the manner in which participants performed the exercises for 20 different test cases. Outlined in this report are my processes for cross validation,building the model, and estimating out of sample error.

### **Data Processing**

```r
# Read test and training data
train_in<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header = T)
validation<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header = T)

# Partition data
set.seed(127)
library(caret)
training_sample <- createDataPartition(y=train_in$classe, p=0.7, list=FALSE)
training <- train_in[training_sample, ]
testing <- train_in[-training_sample, ]

# Identify variables that do not contain zeros
all_zero_colnames <- sapply(names(validation), function(x) all(is.na(validation[,x])==TRUE))
nznames <- names(all_zero_colnames)[all_zero_colnames==FALSE]
nznames <- nznames[-(1:7)]
nznames <- nznames[1:(length(nznames)-1)]
```

### **Modeling**

Cross validation was first performed before modeling the data.

```r
#Cross validation with k = 3
fitControl <- trainControl(method='cv', number = 3)
```


Three modeling techniques were then applied: Decision Tree, Boosting, and Random Forest

```r
library(rpart)
library(caret)
library(rattle)
model_cart <- train(
  classe ~ ., 
  data=training[, c('classe', nznames)],
  trControl=fitControl,
  method='rpart'
)
save(model_cart, file='./ModelFitCART.RData')
model_gbm <- train(
  classe ~ ., 
  data=training[, c('classe', nznames)],
  trControl=fitControl,
  method='gbm'
)
```

```
Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1289
     2        1.5227             nan     0.1000    0.0863
     3        1.4643             nan     0.1000    0.0659
     4        1.4203             nan     0.1000    0.0563
     5        1.3830             nan     0.1000    0.0514
     6        1.3496             nan     0.1000    0.0439
     7        1.3203             nan     0.1000    0.0351
     8        1.2967             nan     0.1000    0.0299
     9        1.2763             nan     0.1000    0.0372
    10        1.2519             nan     0.1000    0.0324
    20        1.0961             nan     0.1000    0.0201
    40        0.9180             nan     0.1000    0.0111
    60        0.8096             nan     0.1000    0.0063
    80        0.7270             nan     0.1000    0.0039
   100        0.6634             nan     0.1000    0.0035
   120        0.6130             nan     0.1000    0.0031
   140        0.5706             nan     0.1000    0.0027
   150        0.5503             nan     0.1000    0.0020

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1932
     2        1.4857             nan     0.1000    0.1326
     3        1.4021             nan     0.1000    0.1044
     4        1.3346             nan     0.1000    0.0822
     5        1.2809             nan     0.1000    0.0753
     6        1.2328             nan     0.1000    0.0626
     7        1.1927             nan     0.1000    0.0644
     8        1.1521             nan     0.1000    0.0566
     9        1.1159             nan     0.1000    0.0487
    10        1.0856             nan     0.1000    0.0419
    20        0.8846             nan     0.1000    0.0199
    40        0.6651             nan     0.1000    0.0098
    60        0.5405             nan     0.1000    0.0081
    80        0.4510             nan     0.1000    0.0053
   100        0.3846             nan     0.1000    0.0036
   120        0.3328             nan     0.1000    0.0026
   140        0.2940             nan     0.1000    0.0010
   150        0.2779             nan     0.1000    0.0021

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2346
     2        1.4587             nan     0.1000    0.1582
     3        1.3547             nan     0.1000    0.1326
     4        1.2701             nan     0.1000    0.1171
     5        1.1989             nan     0.1000    0.0764
     6        1.1495             nan     0.1000    0.0791
     7        1.0989             nan     0.1000    0.0651
     8        1.0567             nan     0.1000    0.0561
     9        1.0204             nan     0.1000    0.0594
    10        0.9827             nan     0.1000    0.0495
    20        0.7469             nan     0.1000    0.0293
    40        0.5182             nan     0.1000    0.0118
    60        0.3894             nan     0.1000    0.0050
    80        0.3110             nan     0.1000    0.0043
   100        0.2538             nan     0.1000    0.0014
   120        0.2146             nan     0.1000    0.0017
   140        0.1840             nan     0.1000    0.0020
   150        0.1699             nan     0.1000    0.0014

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1224
     2        1.5239             nan     0.1000    0.0884
     3        1.4653             nan     0.1000    0.0657
     4        1.4211             nan     0.1000    0.0532
     5        1.3854             nan     0.1000    0.0449
     6        1.3545             nan     0.1000    0.0470
     7        1.3242             nan     0.1000    0.0323
     8        1.3026             nan     0.1000    0.0319
     9        1.2814             nan     0.1000    0.0311
    10        1.2610             nan     0.1000    0.0301
    20        1.1031             nan     0.1000    0.0144
    40        0.9314             nan     0.1000    0.0102
    60        0.8217             nan     0.1000    0.0055
    80        0.7416             nan     0.1000    0.0043
   100        0.6782             nan     0.1000    0.0028
   120        0.6272             nan     0.1000    0.0017
   140        0.5837             nan     0.1000    0.0027
   150        0.5643             nan     0.1000    0.0018

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1821
     2        1.4907             nan     0.1000    0.1254
     3        1.4087             nan     0.1000    0.1063
     4        1.3396             nan     0.1000    0.0811
     5        1.2863             nan     0.1000    0.0761
     6        1.2373             nan     0.1000    0.0583
     7        1.1988             nan     0.1000    0.0598
     8        1.1600             nan     0.1000    0.0577
     9        1.1250             nan     0.1000    0.0446
    10        1.0955             nan     0.1000    0.0421
    20        0.8978             nan     0.1000    0.0195
    40        0.6835             nan     0.1000    0.0129
    60        0.5526             nan     0.1000    0.0051
    80        0.4622             nan     0.1000    0.0046
   100        0.4006             nan     0.1000    0.0035
   120        0.3449             nan     0.1000    0.0023
   140        0.3052             nan     0.1000    0.0027
   150        0.2884             nan     0.1000    0.0015

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2392
     2        1.4615             nan     0.1000    0.1600
     3        1.3601             nan     0.1000    0.1183
     4        1.2834             nan     0.1000    0.1048
     5        1.2158             nan     0.1000    0.0786
     6        1.1639             nan     0.1000    0.0771
     7        1.1152             nan     0.1000    0.0772
     8        1.0673             nan     0.1000    0.0578
     9        1.0304             nan     0.1000    0.0583
    10        0.9912             nan     0.1000    0.0556
    20        0.7580             nan     0.1000    0.0263
    40        0.5275             nan     0.1000    0.0093
    60        0.4036             nan     0.1000    0.0064
    80        0.3234             nan     0.1000    0.0030
   100        0.2651             nan     0.1000    0.0023
   120        0.2222             nan     0.1000    0.0019
   140        0.1885             nan     0.1000    0.0021
   150        0.1737             nan     0.1000    0.0012

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1299
     2        1.5230             nan     0.1000    0.0863
     3        1.4640             nan     0.1000    0.0700
     4        1.4186             nan     0.1000    0.0511
     5        1.3833             nan     0.1000    0.0439
     6        1.3543             nan     0.1000    0.0480
     7        1.3240             nan     0.1000    0.0377
     8        1.2995             nan     0.1000    0.0341
     9        1.2774             nan     0.1000    0.0336
    10        1.2549             nan     0.1000    0.0306
    20        1.0956             nan     0.1000    0.0164
    40        0.9218             nan     0.1000    0.0106
    60        0.8139             nan     0.1000    0.0052
    80        0.7344             nan     0.1000    0.0030
   100        0.6709             nan     0.1000    0.0027
   120        0.6188             nan     0.1000    0.0030
   140        0.5744             nan     0.1000    0.0028
   150        0.5549             nan     0.1000    0.0022

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.1819
     2        1.4897             nan     0.1000    0.1400
     3        1.4007             nan     0.1000    0.1072
     4        1.3321             nan     0.1000    0.0841
     5        1.2783             nan     0.1000    0.0695
     6        1.2327             nan     0.1000    0.0723
     7        1.1865             nan     0.1000    0.0582
     8        1.1489             nan     0.1000    0.0480
     9        1.1187             nan     0.1000    0.0470
    10        1.0887             nan     0.1000    0.0495
    20        0.8858             nan     0.1000    0.0245
    40        0.6676             nan     0.1000    0.0074
    60        0.5437             nan     0.1000    0.0057
    80        0.4583             nan     0.1000    0.0061
   100        0.3912             nan     0.1000    0.0033
   120        0.3404             nan     0.1000    0.0018
   140        0.3017             nan     0.1000    0.0027
   150        0.2816             nan     0.1000    0.0011

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2493
     2        1.4559             nan     0.1000    0.1623
     3        1.3550             nan     0.1000    0.1236
     4        1.2770             nan     0.1000    0.1018
     5        1.2099             nan     0.1000    0.0886
     6        1.1547             nan     0.1000    0.0789
     7        1.1030             nan     0.1000    0.0680
     8        1.0595             nan     0.1000    0.0696
     9        1.0172             nan     0.1000    0.0525
    10        0.9837             nan     0.1000    0.0450
    20        0.7446             nan     0.1000    0.0222
    40        0.5267             nan     0.1000    0.0093
    60        0.4018             nan     0.1000    0.0082
    80        0.3195             nan     0.1000    0.0026
   100        0.2631             nan     0.1000    0.0043
   120        0.2202             nan     0.1000    0.0028
   140        0.1865             nan     0.1000    0.0019
   150        0.1734             nan     0.1000    0.0005

Iter   TrainDeviance   ValidDeviance   StepSize   Improve
     1        1.6094             nan     0.1000    0.2424
     2        1.4561             nan     0.1000    0.1640
     3        1.3529             nan     0.1000    0.1264
     4        1.2734             nan     0.1000    0.0973
     5        1.2112             nan     0.1000    0.0917
     6        1.1539             nan     0.1000    0.0856
     7        1.1006             nan     0.1000    0.0725
     8        1.0535             nan     0.1000    0.0590
     9        1.0154             nan     0.1000    0.0501
    10        0.9833             nan     0.1000    0.0448
    20        0.7581             nan     0.1000    0.0280
    40        0.5230             nan     0.1000    0.0106
    60        0.4042             nan     0.1000    0.0050
    80        0.3232             nan     0.1000    0.0037
   100        0.2666             nan     0.1000    0.0024
   120        0.2238             nan     0.1000    0.0019
   140        0.1914             nan     0.1000    0.0022
   150        0.1773             nan     0.1000    0.0010
```

```r
save(model_gbm, file='./ModelFitGBM.RData')
model_rf <- train(
  classe ~ ., 
  data=training[, c('classe', nznames)],
  trControl=fitControl,
  method='rf',
  ntree=100
)
save(model_rf, file='./ModelFitRF.RData')

champModel<-model_rf
```
### **Model Performance**
The accuracy rate of each model was calculated for comparison.

```r
predCART <- predict(model_cart, newdata=testing)
cmCART <- confusionMatrix(predCART, testing$classe)
predGBM <- predict(model_gbm, newdata=testing)
cmGBM <- confusionMatrix(predGBM, testing$classe)
predRF <- predict(model_rf, newdata=testing)
cmRF <- confusionMatrix(predRF, testing$classe)
AccuracyResults <- data.frame(
  Model = c('CART', 'GBM', 'RF'),
  Accuracy = rbind(cmCART$overall[1], cmGBM$overall[1], cmRF$overall[1])
)
print(AccuracyResults)
```

```
  Model  Accuracy
1  CART 0.4936279
2   GBM 0.9631266
3    RF 0.9938828
```


### **Conclusion**
Per the accuracy rates above, gradient boosting and random forest both outperform the decision tree with random forest being the best model overall. The random forest model's superiority was further confirmed by its ability to predict all 20 test cases correctly.  


```r
predValidation <- predict(champModel, newdata=validation)
ValidationPredictionResults <- data.frame(
  problem_id=validation$problem_id,
  predicted=predValidation
)
print(ValidationPredictionResults)
```

```
   problem_id predicted
1           1         B
2           2         A
3           3         B
4           4         A
5           5         A
6           6         E
7           7         D
8           8         B
9           9         A
10         10         A
11         11         B
12         12         C
13         13         B
14         14         A
15         15         E
16         16         E
17         17         A
18         18         B
19         19         B
20         20         B
```


