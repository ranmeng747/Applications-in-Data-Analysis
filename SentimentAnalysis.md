Sentiment Analysis
================
Ran Meng
March 15, 2019

Set working directory & load environment to avoid retraining when knit:

``` r
setwd("C:/Ran/Berkeley/IEOR/242/Applications in Data Analysis")
load(file="hw4.RData")
```

    ## Warning in readChar(con, 5L, useBytes = TRUE): cannot open compressed file
    ## 'hw4.RData', probable reason 'No such file or directory'

    ## Error in readChar(con, 5L, useBytes = TRUE): cannot open the connection

Load packages

``` r
library(tm)
library(SnowballC)
library(wordcloud)
library(MASS)
library(caTools)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(tm.plugin.webmining)
library(boot)
```

a:
--

i: We start reading the csv, inspect our data:

``` r
sof = read.csv("ggplot2questions2016_17.csv", stringsAsFactors=FALSE)
str(sof)
```

    ## 'data.frame':    7468 obs. of  3 variables:
    ##  $ Title: chr  "Missing Ribbon in ggplot2" "ggplot - label as calculated increase (%)" "Add legend to a ggplot2 from empty plot" "Using ggplot to plot occupied parking spaces in a parking lot" ...
    ##  $ Body : chr  "<p>I seem to be having trouble setting up a ribbon in ggplot2 to display. </p>\n\n<p>Here's a made up data set:"| __truncated__ "<p>I have developed a ggplot graph but now I am trying to add calculated label which shows increase in % year-o"| __truncated__ "<p>I am creating a circular plot of correlations and I want to draw two type of information:</p>\n\n<ol>\n<li>A"| __truncated__ "<p>Iâ\200\231d like to use ggplot to draw a grid plot of the following scenario which Iâ\200\231ve attempted to"| __truncated__ ...
    ##  $ Score: int  3 3 3 6 1 -1 3 14 3 2 ...

ii & iii:

By inspecting the structure, we observe that the content of *Body* feature is HTML content containing tags. We first need to reformat the HTML content to regular text format by removing the tags:

``` r
for (i in 1:length(sof$Body)){
  sof$Body[i] <- extractHTMLStrip(sof$Body[i])
}
head(sof$Body, 2)
```

    ## [1] "I seem to be having trouble setting up a ribbon in ggplot2 to display. \n\nHere's a made up data set:\n\n> GlobalDFData\n  Estimate Upper  Lower  Date   Area\n1      100   125    75 Q1_16 Global\n2      125   150   100 Q2_16 Global\n3      150   175   125 Q3_16 Global\n4      175   200   150 Q4_16 Global\n\n\nHere's the code that I'm trying with no success. I get the line chart but not the upper and lower bounds\n\nggplot(GlobalDFData, aes(x = Date)) + \n  geom_line(aes(y = Estimate, group = Area, color = Area))+\n  geom_point(aes(y = Estimate, x = Date))+\n  geom_ribbon(aes(ymin = Lower, ymax = Upper))\n"                                                                                                                              
    ## [2] "I have developed a ggplot graph but now I am trying to add calculated label which shows increase in % year-on-year?\n\nMy data frame is very simple (result of aggregate from the main dataset)\n\n'data.frame':   4 obs. of  3 variables:\n $ Year    : int  2011 2012 2013 2014\n $ TotalPay: num  71744 74113 77611 75466\n\n\nI have a code for my graph:\n\nlibrary(ggplot2)\nggplot(d1, aes(x=Year, y=TotalPay)) + geom_bar(stat=\"identity\") + \nlabs(x=\"Year\", y=\"Average Total Pay ($)\")\n\n\nand now trying to use stat_bin for lables? The calculation is Actual Year - Previous Year * 100%. I have this but not sure how to fill percent ()\n\nstat_bin(aes (labels = paste(\"Total Pay\" = ,scales::percent(())), vjust = 1, geom = \"TexT\")\n"

Reconstruct our dataframe by labelling whether the question is useful (score &gt;= 1)according to project specification:

``` r
sof$useful = as.factor(as.numeric(sof$Score >= 1))
sof$Score <- NULL # No longer needed
table(sof$useful)
```

    ## 
    ##    0    1 
    ## 3791 3677

From the table above, the majority class is 0, meaning the baseline model is predicting all questions as *not useful*. iv & v:

Now we need to convert our title & body content to corpus as the next step to prepare for our analysis:

``` r
corpus_title = Corpus(VectorSource(sof$Title))
corpus_body = Corpus(VectorSource(sof$Body))

strwrap(corpus_title[[1]])
```

    ## [1] "Missing Ribbon in ggplot2"

``` r
strwrap(corpus_body[[1]])
```

    ##  [1] "I seem to be having trouble setting up a ribbon in ggplot2 to"     
    ##  [2] "display."                                                          
    ##  [3] ""                                                                  
    ##  [4] "Here's a made up data set:"                                        
    ##  [5] ""                                                                  
    ##  [6] "> GlobalDFData Estimate Upper Lower Date Area 1 100 125 75 Q1_16"  
    ##  [7] "Global 2 125 150 100 Q2_16 Global 3 150 175 125 Q3_16 Global 4 175"
    ##  [8] "200 150 Q4_16 Global"                                              
    ##  [9] ""                                                                  
    ## [10] "Here's the code that I'm trying with no success. I get the line"   
    ## [11] "chart but not the upper and lower bounds"                          
    ## [12] ""                                                                  
    ## [13] "ggplot(GlobalDFData, aes(x = Date)) + geom_line(aes(y = Estimate," 
    ## [14] "group = Area, color = Area))+ geom_point(aes(y = Estimate, x ="    
    ## [15] "Date))+ geom_ribbon(aes(ymin = Lower, ymax = Upper))"

Now we have removed the HTML elements and transformed our content to corpus documents, the next step is to clean up irregularities so that words that contain the same meaning are actually attributed together. We can start by transforming all the content to lower cases:

``` r
corpus_title = tm_map(corpus_title, tolower)
corpus_body = tm_map(corpus_body, tolower)
strwrap(corpus_body[[1]])
```

    ##  [1] "i seem to be having trouble setting up a ribbon in ggplot2 to"     
    ##  [2] "display."                                                          
    ##  [3] ""                                                                  
    ##  [4] "here's a made up data set:"                                        
    ##  [5] ""                                                                  
    ##  [6] "> globaldfdata estimate upper lower date area 1 100 125 75 q1_16"  
    ##  [7] "global 2 125 150 100 q2_16 global 3 150 175 125 q3_16 global 4 175"
    ##  [8] "200 150 q4_16 global"                                              
    ##  [9] ""                                                                  
    ## [10] "here's the code that i'm trying with no success. i get the line"   
    ## [11] "chart but not the upper and lower bounds"                          
    ## [12] ""                                                                  
    ## [13] "ggplot(globaldfdata, aes(x = date)) + geom_line(aes(y = estimate," 
    ## [14] "group = area, color = area))+ geom_point(aes(y = estimate, x ="    
    ## [15] "date))+ geom_ribbon(aes(ymin = lower, ymax = upper))"

I decide not to remove the punctations for now because people often post code on Stack OverFlows. I think removing punctuations like "()", "\[\]" and "&lt;-" will make it hard to distunguish the posts that contain R code and the posts that do not. From my intuition, posts that contain code may have higher usefulness ratings. Nevertheless, I will remove punctuations such as "," and ".", which are less commonly used in R language.

``` r
#corpus_body = tm_map(corpus_body, removePunctuation)
#corpus_title = tm_map(corpus_title, removePunctuation)
#strwrap(corpus_body[[1]])
```

I will further pre-process the data by removing common English stopwords (words that are not useful for analysis), as well as "ggplot/ggplot2" since all threads are about the package " ggplot/ggplot2" so it is a word that is common to all observations

``` r
corpus_title = tm_map(corpus_title, removeWords, c("ggplot2", "ggplot", stopwords("english")))
corpus_body = tm_map(corpus_body, removeWords, c("ggplot2", stopwords("english")))
strwrap(corpus_body[[1]])
```

    ##  [1] "seem trouble setting ribbon display."                              
    ##  [2] ""                                                                  
    ##  [3] "made data set:"                                                    
    ##  [4] ""                                                                  
    ##  [5] "> globaldfdata estimate upper lower date area 1 100 125 75 q1_16"  
    ##  [6] "global 2 125 150 100 q2_16 global 3 150 175 125 q3_16 global 4 175"
    ##  [7] "200 150 q4_16 global"                                              
    ##  [8] ""                                                                  
    ##  [9] "code trying success.  get line chart upper lower bounds"           
    ## [10] ""                                                                  
    ## [11] "ggplot(globaldfdata, aes(x = date)) + geom_line(aes(y = estimate," 
    ## [12] "group = area, color = area))+ geom_point(aes(y = estimate, x ="    
    ## [13] "date))+ geom_ribbon(aes(ymin = lower, ymax = upper))"

Let us stem the words to avoid trying distinctions between same word under different tenses:

``` r
corpus_title = tm_map(corpus_title, stemDocument)
corpus_body = tm_map(corpus_body, stemDocument)
strwrap(corpus_body[[1]])
```

    ## [1] "seem troubl set ribbon display. made data set: > globaldfdata"    
    ## [2] "estim upper lower date area 1 100 125 75 q1_16 global 2 125 150"  
    ## [3] "100 q2_16 global 3 150 175 125 q3_16 global 4 175 200 150 q4_16"  
    ## [4] "global code tri success. get line chart upper lower bound"        
    ## [5] "ggplot(globaldfdata, aes(x = date)) + geom_line(aes(i = estimate,"
    ## [6] "group = area, color = area))+ geom_point(aes(i = estimate, x ="   
    ## [7] "date))+ geom_ribbon(aes(ymin = lower, ymax = upper))"

vi:

Implied from Zipf's law, the majority of the text will be uncommon words that do not appear many times. We should filter out the uncommon words as part of our *bag of words* analysis because without doing so, the data size will be too large for regression models training. Another consideration that we need to take is the time& space complexity. Keeping the uncommon words in training data will add up the training time & space significantly if we were training with tree- based models such as RF or Boosting. The drawback of this approach is that we may lose some information. As a reuslt, there may be a tradeoff between practicality and accuracy of our models. Finding a balance with the right parameters is thus important.

To start the process of removing the "uncommon" (we need to define this ourselves) words, we need to construct a word count matrix:

``` r
frequencies_title = DocumentTermMatrix(corpus_title)
frequencies_body = DocumentTermMatrix(corpus_body)
```

Frequencies of titles:

``` r
frequencies_title
```

    ## <<DocumentTermMatrix (documents: 7468, terms: 4845)>>
    ## Non-/sparse entries: 37765/36144695
    ## Sparsity           : 100%
    ## Maximal term length: 68
    ## Weighting          : term frequency (tf)

Frequencies of bodies:

``` r
frequencies_body
```

    ## <<DocumentTermMatrix (documents: 7468, terms: 252901)>>
    ## Non-/sparse entries: 686965/1887977703
    ## Sparsity           : 100%
    ## Maximal term length: 2909
    ## Weighting          : term frequency (tf)

By inspecting *terms*, we can observe 4870 independent variables for *title*, and 2527269 for *body*. Given the time & space constraint given above, I want to limit my total number of independent variables to ~100 for starting point(subject to change when I actually start building models). So let us find out what happens:

Let us begin by keeping only the terms that appear in 1% of the posts or more, and see what are the terms:

``` r
sparse_title <- removeSparseTerms(frequencies_title, 0.99)
sparse_title 
```

    ## <<DocumentTermMatrix (documents: 7468, terms: 79)>>
    ## Non-/sparse entries: 18570/571402
    ## Sparsity           : 97%
    ## Maximal term length: 10
    ## Weighting          : term frequency (tf)

``` r
head(sparse_title$dimnames$Terms, 20) 
```

    ##  [1] "label"    "add"      "legend"   "plot"     "space"    "use"     
    ##  [7] "object"   "error"    "group"    "axi"      "can"      "chang"   
    ## [13] "map"      "data"     "function" "graph"    "make"     "line"    
    ## [19] "multipl"  "display"

``` r
sparse_body<- removeSparseTerms(frequencies_body, 0.99)
sparse_body 
```

    ## <<DocumentTermMatrix (documents: 7468, terms: 918)>>
    ## Non-/sparse entries: 255965/6599659
    ## Sparsity           : 96%
    ## Maximal term length: 25
    ## Weighting          : term frequency (tf)

``` r
head(sparse_body$dimnames$Terms, 20)
```

    ##  [1] "100"    "aes(x"  "area"   "chart"  "code"   "color"  "data"  
    ##  [8] "date"   "estim"  "get"    "group"  "line"   "lower"  "made"  
    ## [15] "seem"   "set"    "tri"    "troubl" "upper"  "ymax"

Unfortunately there is a total of ~1000 variables, way more than what I think my computation resource can handle. So let us remove more "Uncommon" words, this time separately:

Using different sparse parameters:

``` r
sparse_title <- removeSparseTerms(frequencies_title, 0.98)
sparse_title 
```

    ## <<DocumentTermMatrix (documents: 7468, terms: 44)>>
    ## Non-/sparse entries: 15188/313404
    ## Sparsity           : 95%
    ## Maximal term length: 9
    ## Weighting          : term frequency (tf)

``` r
head(sparse_title$dimnames$Terms, 20) 
```

    ##  [1] "label"    "add"      "legend"   "plot"     "use"      "error"   
    ##  [7] "group"    "axi"      "can"      "chang"    "map"      "data"    
    ## [13] "function" "graph"    "make"     "line"     "multipl"  "time"    
    ## [19] "boxplot"  "factor"

``` r
sparse_body<- removeSparseTerms(frequencies_body, 0.94)
sparse_body 
```

    ## <<DocumentTermMatrix (documents: 7468, terms: 114)>>
    ## Non-/sparse entries: 122077/729275
    ## Sparsity           : 86%
    ## Maximal term length: 12
    ## Weighting          : term frequency (tf)

``` r
head(sparse_body$dimnames$Terms, 20)
```

    ##  [1] "aes(x"     "chart"     "code"      "color"     "data"     
    ##  [6] "get"       "group"     "line"      "seem"      "set"      
    ## [11] "tri"       "add"       "fill"      "frame"     "ggplot"   
    ## [16] "graph"     "label"     "library()" "now"       "show"

We have cut down the features quite a bit, let us now construct a dataframe table and group the features that are unique:

``` r
sofTM_title = as.data.frame(as.matrix(sparse_title))
sofTM_body = as.data.frame(as.matrix(sparse_body))
sofTM = cbind(sofTM_title, sofTM_body)
sofTM <- sapply(unique(colnames(sofTM)), 
       function(x) rowSums(sofTM[, colnames(sofTM) == x, drop = FALSE])) # combine unique features and add their values
#Convert back to df
sofTM <- as.data.frame(sofTM)
dim(sofTM)
```

    ## [1] 7468  120

Since some of the independent variables are numbers, we need to rename the column names that are accepted by R:

``` r
colnames(sofTM) = make.names(colnames(sofTM))
```

Adding dependent variable:

``` r
sofTM <- cbind(sofTM, useful = sof$useful) #Not sure why the original expression gives me 1s and 2s
```

Dimension of final, processed dataset:

``` r
dim(sofTM)
```

    ## [1] 7468  121

b:
--

Split data into train& test. We need to use a random split to make sure that the relative amount of useful questions is approximately the same :

``` r
set.seed(123)  
spl = sample.split(sofTM$useful, SplitRatio = 0.7)

sofTrain = sofTM %>% filter(spl == TRUE)
sofTest = sofTM %>% filter(spl == FALSE)

table(sofTrain$useful)
```

    ## 
    ##    0    1 
    ## 2654 2574

``` r
table(sofTest$useful)
```

    ## 
    ##    0    1 
    ## 1137 1103

``` r
cat("Accuracy of baseline model is ", table(sofTest$useful)[2]/length(sofTest$useful))
```

    ## Accuracy of baseline model is  0.4924107

From the tables, we can see that the number of *not useful* questions is slightly higher than the number of *useful* questions for both the train and test data, meaning we have a balanced dataset.

Before making predictions, let us create a function that will help us calculating the prediction accuracy, TPR and FPR:

``` r
tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}

tableTPR <- function(test, pred){
  t = table(test, pred)
  tpr = t[4]/(t[4] + t[2])
  return (tpr)
}

tableFPR <- function(test, pred){
  t = table(test, pred)
  fpr = t[3]/(t[3] + t[1])
  return (fpr)
}
```

### b1: Basic Logstic Regression

To predict, let us start with the **basic Logistic Model**:

``` r
sof_log = glm(useful ~ ., data = sofTrain, family = "binomial")
summary(sof_log)
```

    ## 
    ## Call:
    ## glm(formula = useful ~ ., family = "binomial", data = sofTrain)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.1278  -1.1035  -0.7179   1.1498   2.5136  
    ## 
    ## Coefficients:
    ##                Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  -3.294e-01  6.650e-02  -4.954 7.28e-07 ***
    ## label         3.439e-02  2.611e-02   1.317 0.187817    
    ## add           1.551e-03  4.302e-02   0.036 0.971236    
    ## legend        3.390e-02  3.200e-02   1.059 0.289386    
    ## plot         -3.409e-02  1.605e-02  -2.124 0.033692 *  
    ## use           2.484e-02  2.425e-02   1.024 0.305683    
    ## error        -9.930e-02  4.288e-02  -2.316 0.020577 *  
    ## group        -3.774e-02  3.267e-02  -1.155 0.247954    
    ## axi          -4.895e-03  4.231e-02  -0.116 0.907896    
    ## can           3.886e-02  3.686e-02   1.054 0.291788    
    ## chang         5.370e-03  4.355e-02   0.123 0.901868    
    ## map           2.612e-02  3.434e-02   0.761 0.446849    
    ## data         -7.311e-03  2.281e-02  -0.321 0.748535    
    ## function.     5.186e-02  3.559e-02   1.457 0.145104    
    ## graph        -4.066e-02  3.439e-02  -1.182 0.237137    
    ## make          4.036e-02  4.629e-02   0.872 0.383251    
    ## line          4.371e-03  2.737e-02   0.160 0.873104    
    ## multipl      -1.132e-01  6.254e-02  -1.811 0.070183 .  
    ## time         -5.668e-02  4.147e-02  -1.367 0.171684    
    ## boxplot      -3.056e-01  2.173e-01  -1.406 0.159657    
    ## factor       -9.304e-03  5.484e-02  -0.170 0.865289    
    ## facet         8.821e-02  4.257e-02   2.072 0.038231 *  
    ## histogram    -1.401e-01  1.956e-01  -0.717 0.473642    
    ## column       -1.934e-01  4.846e-02  -3.992 6.56e-05 ***
    ## creat         7.085e-03  3.922e-02   0.181 0.856645    
    ## shini        -1.611e-01  2.093e-01  -0.770 0.441519    
    ## colour        1.249e-02  3.434e-02   0.364 0.716161    
    ## color        -7.415e-03  2.509e-02  -0.296 0.767604    
    ## two          -1.708e-02  3.761e-02  -0.454 0.649698    
    ## x.axi        -6.515e-02  6.554e-02  -0.994 0.320235    
    ## scale         6.566e-02  4.856e-02   1.352 0.176356    
    ## text          1.975e-01  1.936e-01   1.020 0.307717    
    ## variabl      -5.966e-02  3.697e-02  -1.614 0.106534    
    ## valu         -5.648e-02  2.787e-02  -2.026 0.042756 *  
    ## set           5.486e-03  5.110e-02   0.107 0.914515    
    ## point         4.747e-02  3.409e-02   1.392 0.163829    
    ## differ       -2.319e-02  3.979e-02  -0.583 0.559952    
    ## bar          -1.167e-02  2.951e-02  -0.396 0.692411    
    ## barplot       8.400e-03  1.960e-01   0.043 0.965808    
    ## show          5.776e-02  4.843e-02   1.193 0.233015    
    ## one          -3.215e-02  4.093e-02  -0.785 0.432181    
    ## stack         1.771e-01  1.646e-01   1.076 0.281909    
    ## order        -5.914e-03  4.811e-02  -0.123 0.902177    
    ## fill         -3.409e-02  3.920e-02  -0.870 0.384524    
    ## chart        -2.599e-02  4.824e-02  -0.539 0.590002    
    ## aes.x         2.928e-02  4.524e-02   0.647 0.517523    
    ## code         -7.862e-02  4.026e-02  -1.953 0.050829 .  
    ## get           1.275e-02  4.123e-02   0.309 0.757201    
    ## seem          1.221e-01  7.777e-02   1.570 0.116462    
    ## tri           5.090e-03  3.780e-02   0.135 0.892877    
    ## frame         3.280e-02  6.780e-02   0.484 0.628576    
    ## ggplot       -2.224e-02  4.697e-02  -0.474 0.635803    
    ## library..     3.127e-01  6.337e-02   4.934 8.04e-07 ***
    ## now           1.545e-01  7.797e-02   1.982 0.047474 *  
    ## dataset      -6.101e-02  6.935e-02  -0.880 0.378999    
    ## exampl        2.118e-01  6.112e-02   3.465 0.000530 ***
    ## first        -1.074e-01  7.212e-02  -1.489 0.136510    
    ## like          7.799e-02  3.346e-02   2.330 0.019781 *  
    ## second        1.797e-01  1.023e-01   1.757 0.078969 .  
    ## see           7.545e-02  7.837e-02   0.963 0.335643    
    ## size          3.708e-02  3.086e-02   1.201 0.229594    
    ## solut         3.702e-01  9.664e-02   3.830 0.000128 ***
    ## type         -6.475e-03  6.285e-02  -0.103 0.917947    
    ## want         -8.504e-03  3.736e-02  -0.228 0.819948    
    ## follow        7.154e-02  4.850e-02   1.475 0.140175    
    ## look         -4.765e-02  5.241e-02  -0.909 0.363252    
    ## number        8.319e-02  6.235e-02   1.334 0.182149    
    ## someth        1.612e-01  8.593e-02   1.876 0.060661 .  
    ## thank        -1.950e-01  7.644e-02  -2.550 0.010757 *  
    ## also          1.252e-01  7.409e-02   1.690 0.091071 .  
    ## found        -4.833e-05  1.041e-01   0.000 0.999630    
    ## however.      5.263e-02  7.347e-02   0.716 0.473759    
    ## mean          3.307e-02  4.193e-02   0.789 0.430282    
    ## object        1.407e-02  8.046e-02   0.175 0.861161    
    ## produc        1.650e-01  7.022e-02   2.349 0.018821 *  
    ## instead      -9.477e-02  8.394e-02  -1.129 0.258882    
    ## will         -4.548e-03  6.202e-02  -0.073 0.941540    
    ## work         -1.558e-02  5.143e-02  -0.303 0.761928    
    ## plot.         5.447e-02  8.975e-02   0.607 0.543912    
    ## display       1.079e-01  7.990e-02   1.350 0.177015    
    ## help         -7.734e-02  7.770e-02  -0.995 0.319585    
    ## just          1.038e-01  7.111e-02   1.460 0.144369    
    ## possibl       2.417e-01  9.753e-02   2.478 0.013215 *  
    ## sampl         2.286e-03  4.295e-02   0.053 0.957562    
    ## give         -6.829e-02  8.162e-02  -0.837 0.402719    
    ## .name        -8.749e-02  7.828e-02  -1.118 0.263702    
    ## c.na.        -6.984e-02  1.263e-01  -0.553 0.580410    
    ## class         4.596e-03  4.489e-02   0.102 0.918458    
    ## row.nam       2.216e-01  1.672e-01   1.325 0.185162    
    ## code.        -9.117e-02  8.550e-02  -1.066 0.286249    
    ## know         -1.184e-01  6.935e-02  -1.707 0.087854 .  
    ## error.       -1.066e-01  8.364e-02  -1.275 0.202394    
    ## width         7.748e-02  6.175e-02   1.255 0.209531    
    ## abl          -8.281e-02  8.800e-02  -0.941 0.346647    
    ## data.         7.008e-02  1.104e-01   0.635 0.525490    
    ## figur         2.442e-02  6.990e-02   0.349 0.726828    
    ## generat      -1.048e-02  7.020e-02  -0.149 0.881277    
    ## ggplot.data   3.118e-03  7.714e-02   0.040 0.967757    
    ## level         5.567e-03  5.949e-02   0.094 0.925439    
    ## new          -1.311e-02  7.238e-02  -0.181 0.856308    
    ## problem      -9.536e-02  6.786e-02  -1.405 0.159925    
    ## question      1.482e-01  7.690e-02   1.927 0.053979 .  
    ## ggplot.df.    2.716e-02  7.972e-02   0.341 0.733332    
    ## packag        2.064e-02  6.307e-02   0.327 0.743442    
    ## find          3.597e-02  8.723e-02   0.412 0.680047    
    ## run           1.262e-01  7.794e-02   1.619 0.105349    
    ## way          -4.103e-03  5.576e-02  -0.074 0.941340    
    ## need         -4.820e-02  6.145e-02  -0.784 0.432855    
    ## output        1.793e-01  8.458e-02   2.119 0.034052 *  
    ## geom_point..  1.636e-01  6.618e-02   2.473 0.013414 *  
    ## base          1.473e-01  6.531e-02   2.256 0.024094 *  
    ## name         -7.398e-02  5.932e-02  -1.247 0.212317    
    ## similar       2.384e-01  1.072e-01   2.225 0.026111 *  
    ## anyon         5.744e-02  1.157e-01   0.497 0.619503    
    ## result        1.382e-01  7.514e-02   1.839 0.065961 .  
    ## theme_bw..    8.281e-02  8.785e-02   0.943 0.345879    
    ## right         1.048e-01  8.528e-02   1.229 0.218974    
    ## X...          2.108e-02  2.482e-02   0.849 0.395758    
    ## without       5.910e-02  9.658e-02   0.612 0.540577    
    ## posit         2.895e-02  5.828e-02   0.497 0.619426    
    ## datafram      5.267e-03  8.670e-02   0.061 0.951560    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 7246.3  on 5227  degrees of freedom
    ## Residual deviance: 6923.6  on 5107  degrees of freedom
    ## AIC: 7165.6
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
# Predictions on test set
predict_log = predict(sof_log, newdata = sofTest, type = "response")
table(sofTest$useful, PredictLog > 0.5)
```

    ## Error in table(sofTest$useful, PredictLog > 0.5): object 'PredictLog' not found

``` r
acc_log <- tableAccuracy(sofTest$useful, predict_log > 0.5)
acc_log
```

    ## [1] 0.56875

``` r
tpr_log <- tableTPR(sofTest$useful, predict_log > 0.5)
tpr_log
```

    ## [1] 0.5049864

``` r
fpr_log <- tableFPR(sofTest$useful, predict_log > 0.5)
fpr_log
```

    ## [1] 0.3693931

The accuracy of the default log model is 0.569, while TPR = 0.505 and FPR = 0.369.

### b2: Cross- validated CART

Would a **CART method with cross validation** work better?

I am trying to construct this CART model with cp = (0, 0.001, 0.002, ..., 0.04) and see which one is the best with a 10-fold cross validation. I choose 10 as fold size because it is a good balance betweem variance and bias. If k is large, there will be less bias but higher correlation. In other words, LOOCV, an extreme case of high k, has more accurate estimates on average but the result is very dependent on the particular training set. I consider 10 as a good compromise and a suitable fold number for this cross-validation. I will pick the cp value with the highest accuracy through the 10-fold cv:

``` r
set.seed(3421)
train.cart = train(useful ~ .,
                   data = sofTrain,
                   method = "rpart",
                   tuneGrid = data.frame(cp=seq(0, 0.4, 0.001)),
                   trControl = trainControl(method="cv", number= 10))
```

Result of training shows that cp = 0.012 gives us the best model:

``` r
train.cart
```

    ## CART 
    ## 
    ## 5228 samples
    ##  120 predictor
    ##    2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4704, 4706, 4705, 4705, 4706, 4706, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp     Accuracy   Kappa       
    ##   0.000  0.5160627   0.031422285
    ##   0.001  0.5202637   0.038987611
    ##   0.002  0.5370992   0.074622173
    ##   0.003  0.5462800   0.091617630
    ##   0.004  0.5497213   0.097198524
    ##   0.005  0.5518264   0.101056783
    ##   0.006  0.5527824   0.103095230
    ##   0.007  0.5539296   0.105357109
    ##   0.008  0.5541219   0.105790945
    ##   0.009  0.5543135   0.106260417
    ##   0.010  0.5548871   0.107468780
    ##   0.011  0.5566113   0.110831311
    ##   0.012  0.5566113   0.110831311
    ##   0.013  0.5560395   0.109435324
    ##   0.014  0.5560395   0.109435324
    ##   0.015  0.5556571   0.108498649
    ##   0.016  0.5533626   0.103274565
    ##   0.017  0.5533626   0.103274565
    ##   0.018  0.5516366   0.099425969
    ##   0.019  0.5516366   0.099425969
    ##   0.020  0.5516366   0.099425969
    ##   0.021  0.5516366   0.099425969
    ##   0.022  0.5516366   0.099425969
    ##   0.023  0.5525930   0.100943527
    ##   0.024  0.5497195   0.094779809
    ##   0.025  0.5472338   0.089465612
    ##   0.026  0.5455162   0.085690744
    ##   0.027  0.5441778   0.082770506
    ##   0.028  0.5432218   0.080531180
    ##   0.029  0.5424570   0.078703319
    ##   0.030  0.5424570   0.078703319
    ##   0.031  0.5426482   0.078873708
    ##   0.032  0.5426482   0.078873708
    ##   0.033  0.5426482   0.078873708
    ##   0.034  0.5426482   0.078873708
    ##   0.035  0.5426482   0.078873708
    ##   0.036  0.5426482   0.078873708
    ##   0.037  0.5426482   0.078873708
    ##   0.038  0.5378589   0.068604916
    ##   0.039  0.5378589   0.068604916
    ##   0.040  0.5378589   0.068604916
    ##   0.041  0.5378589   0.068604916
    ##   0.042  0.5378589   0.068604916
    ##   0.043  0.5378589   0.068604916
    ##   0.044  0.5378589   0.068604916
    ##   0.045  0.5378589   0.068604916
    ##   0.046  0.5378589   0.068604916
    ##   0.047  0.5378589   0.068604916
    ##   0.048  0.5378589   0.068604916
    ##   0.049  0.5378589   0.068604916
    ##   0.050  0.5378589   0.068604916
    ##   0.051  0.5378589   0.068604916
    ##   0.052  0.5378589   0.068604916
    ##   0.053  0.5378589   0.068604916
    ##   0.054  0.5378589   0.068604916
    ##   0.055  0.5378589   0.068604916
    ##   0.056  0.5378589   0.068604916
    ##   0.057  0.5378589   0.068604916
    ##   0.058  0.5378589   0.068604916
    ##   0.059  0.5378589   0.068604916
    ##   0.060  0.5361348   0.064661902
    ##   0.061  0.5361348   0.064661902
    ##   0.062  0.5361348   0.064661902
    ##   0.063  0.5361348   0.064661902
    ##   0.064  0.5361348   0.064661902
    ##   0.065  0.5361348   0.064661902
    ##   0.066  0.5361348   0.064661902
    ##   0.067  0.5361348   0.064661902
    ##   0.068  0.5361348   0.064661902
    ##   0.069  0.5361348   0.064661902
    ##   0.070  0.5361348   0.064661902
    ##   0.071  0.5309722   0.053574096
    ##   0.072  0.5263833   0.043381671
    ##   0.073  0.5221768   0.034020245
    ##   0.074  0.5149180   0.017618068
    ##   0.075  0.5082258   0.002770348
    ##   0.076  0.5082258   0.002770348
    ##   0.077  0.5057354  -0.003079654
    ##   0.078  0.5057354  -0.003079654
    ##   0.079  0.5057354  -0.003079654
    ##   0.080  0.5057354  -0.003079654
    ##   0.081  0.5057354  -0.003079654
    ##   0.082  0.5057354  -0.003079654
    ##   0.083  0.5057354  -0.003079654
    ##   0.084  0.5057354  -0.003079654
    ##   0.085  0.5057354  -0.003079654
    ##   0.086  0.5057354  -0.003079654
    ##   0.087  0.5076511   0.000000000
    ##   0.088  0.5076511   0.000000000
    ##   0.089  0.5076511   0.000000000
    ##   0.090  0.5076511   0.000000000
    ##   0.091  0.5076511   0.000000000
    ##   0.092  0.5076511   0.000000000
    ##   0.093  0.5076511   0.000000000
    ##   0.094  0.5076511   0.000000000
    ##   0.095  0.5076511   0.000000000
    ##   0.096  0.5076511   0.000000000
    ##   0.097  0.5076511   0.000000000
    ##   0.098  0.5076511   0.000000000
    ##   0.099  0.5076511   0.000000000
    ##   0.100  0.5076511   0.000000000
    ##   0.101  0.5076511   0.000000000
    ##   0.102  0.5076511   0.000000000
    ##   0.103  0.5076511   0.000000000
    ##   0.104  0.5076511   0.000000000
    ##   0.105  0.5076511   0.000000000
    ##   0.106  0.5076511   0.000000000
    ##   0.107  0.5076511   0.000000000
    ##   0.108  0.5076511   0.000000000
    ##   0.109  0.5076511   0.000000000
    ##   0.110  0.5076511   0.000000000
    ##   0.111  0.5076511   0.000000000
    ##   0.112  0.5076511   0.000000000
    ##   0.113  0.5076511   0.000000000
    ##   0.114  0.5076511   0.000000000
    ##   0.115  0.5076511   0.000000000
    ##   0.116  0.5076511   0.000000000
    ##   0.117  0.5076511   0.000000000
    ##   0.118  0.5076511   0.000000000
    ##   0.119  0.5076511   0.000000000
    ##   0.120  0.5076511   0.000000000
    ##   0.121  0.5076511   0.000000000
    ##   0.122  0.5076511   0.000000000
    ##   0.123  0.5076511   0.000000000
    ##   0.124  0.5076511   0.000000000
    ##   0.125  0.5076511   0.000000000
    ##   0.126  0.5076511   0.000000000
    ##   0.127  0.5076511   0.000000000
    ##   0.128  0.5076511   0.000000000
    ##   0.129  0.5076511   0.000000000
    ##   0.130  0.5076511   0.000000000
    ##   0.131  0.5076511   0.000000000
    ##   0.132  0.5076511   0.000000000
    ##   0.133  0.5076511   0.000000000
    ##   0.134  0.5076511   0.000000000
    ##   0.135  0.5076511   0.000000000
    ##   0.136  0.5076511   0.000000000
    ##   0.137  0.5076511   0.000000000
    ##   0.138  0.5076511   0.000000000
    ##   0.139  0.5076511   0.000000000
    ##   0.140  0.5076511   0.000000000
    ##   0.141  0.5076511   0.000000000
    ##   0.142  0.5076511   0.000000000
    ##   0.143  0.5076511   0.000000000
    ##   0.144  0.5076511   0.000000000
    ##   0.145  0.5076511   0.000000000
    ##   0.146  0.5076511   0.000000000
    ##   0.147  0.5076511   0.000000000
    ##   0.148  0.5076511   0.000000000
    ##   0.149  0.5076511   0.000000000
    ##   0.150  0.5076511   0.000000000
    ##   0.151  0.5076511   0.000000000
    ##   0.152  0.5076511   0.000000000
    ##   0.153  0.5076511   0.000000000
    ##   0.154  0.5076511   0.000000000
    ##   0.155  0.5076511   0.000000000
    ##   0.156  0.5076511   0.000000000
    ##   0.157  0.5076511   0.000000000
    ##   0.158  0.5076511   0.000000000
    ##   0.159  0.5076511   0.000000000
    ##   0.160  0.5076511   0.000000000
    ##   0.161  0.5076511   0.000000000
    ##   0.162  0.5076511   0.000000000
    ##   0.163  0.5076511   0.000000000
    ##   0.164  0.5076511   0.000000000
    ##   0.165  0.5076511   0.000000000
    ##   0.166  0.5076511   0.000000000
    ##   0.167  0.5076511   0.000000000
    ##   0.168  0.5076511   0.000000000
    ##   0.169  0.5076511   0.000000000
    ##   0.170  0.5076511   0.000000000
    ##   0.171  0.5076511   0.000000000
    ##   0.172  0.5076511   0.000000000
    ##   0.173  0.5076511   0.000000000
    ##   0.174  0.5076511   0.000000000
    ##   0.175  0.5076511   0.000000000
    ##   0.176  0.5076511   0.000000000
    ##   0.177  0.5076511   0.000000000
    ##   0.178  0.5076511   0.000000000
    ##   0.179  0.5076511   0.000000000
    ##   0.180  0.5076511   0.000000000
    ##   0.181  0.5076511   0.000000000
    ##   0.182  0.5076511   0.000000000
    ##   0.183  0.5076511   0.000000000
    ##   0.184  0.5076511   0.000000000
    ##   0.185  0.5076511   0.000000000
    ##   0.186  0.5076511   0.000000000
    ##   0.187  0.5076511   0.000000000
    ##   0.188  0.5076511   0.000000000
    ##   0.189  0.5076511   0.000000000
    ##   0.190  0.5076511   0.000000000
    ##   0.191  0.5076511   0.000000000
    ##   0.192  0.5076511   0.000000000
    ##   0.193  0.5076511   0.000000000
    ##   0.194  0.5076511   0.000000000
    ##   0.195  0.5076511   0.000000000
    ##   0.196  0.5076511   0.000000000
    ##   0.197  0.5076511   0.000000000
    ##   0.198  0.5076511   0.000000000
    ##   0.199  0.5076511   0.000000000
    ##   0.200  0.5076511   0.000000000
    ##   0.201  0.5076511   0.000000000
    ##   0.202  0.5076511   0.000000000
    ##   0.203  0.5076511   0.000000000
    ##   0.204  0.5076511   0.000000000
    ##   0.205  0.5076511   0.000000000
    ##   0.206  0.5076511   0.000000000
    ##   0.207  0.5076511   0.000000000
    ##   0.208  0.5076511   0.000000000
    ##   0.209  0.5076511   0.000000000
    ##   0.210  0.5076511   0.000000000
    ##   0.211  0.5076511   0.000000000
    ##   0.212  0.5076511   0.000000000
    ##   0.213  0.5076511   0.000000000
    ##   0.214  0.5076511   0.000000000
    ##   0.215  0.5076511   0.000000000
    ##   0.216  0.5076511   0.000000000
    ##   0.217  0.5076511   0.000000000
    ##   0.218  0.5076511   0.000000000
    ##   0.219  0.5076511   0.000000000
    ##   0.220  0.5076511   0.000000000
    ##   0.221  0.5076511   0.000000000
    ##   0.222  0.5076511   0.000000000
    ##   0.223  0.5076511   0.000000000
    ##   0.224  0.5076511   0.000000000
    ##   0.225  0.5076511   0.000000000
    ##   0.226  0.5076511   0.000000000
    ##   0.227  0.5076511   0.000000000
    ##   0.228  0.5076511   0.000000000
    ##   0.229  0.5076511   0.000000000
    ##   0.230  0.5076511   0.000000000
    ##   0.231  0.5076511   0.000000000
    ##   0.232  0.5076511   0.000000000
    ##   0.233  0.5076511   0.000000000
    ##   0.234  0.5076511   0.000000000
    ##   0.235  0.5076511   0.000000000
    ##   0.236  0.5076511   0.000000000
    ##   0.237  0.5076511   0.000000000
    ##   0.238  0.5076511   0.000000000
    ##   0.239  0.5076511   0.000000000
    ##   0.240  0.5076511   0.000000000
    ##   0.241  0.5076511   0.000000000
    ##   0.242  0.5076511   0.000000000
    ##   0.243  0.5076511   0.000000000
    ##   0.244  0.5076511   0.000000000
    ##   0.245  0.5076511   0.000000000
    ##   0.246  0.5076511   0.000000000
    ##   0.247  0.5076511   0.000000000
    ##   0.248  0.5076511   0.000000000
    ##   0.249  0.5076511   0.000000000
    ##   0.250  0.5076511   0.000000000
    ##   0.251  0.5076511   0.000000000
    ##   0.252  0.5076511   0.000000000
    ##   0.253  0.5076511   0.000000000
    ##   0.254  0.5076511   0.000000000
    ##   0.255  0.5076511   0.000000000
    ##   0.256  0.5076511   0.000000000
    ##   0.257  0.5076511   0.000000000
    ##   0.258  0.5076511   0.000000000
    ##   0.259  0.5076511   0.000000000
    ##   0.260  0.5076511   0.000000000
    ##   0.261  0.5076511   0.000000000
    ##   0.262  0.5076511   0.000000000
    ##   0.263  0.5076511   0.000000000
    ##   0.264  0.5076511   0.000000000
    ##   0.265  0.5076511   0.000000000
    ##   0.266  0.5076511   0.000000000
    ##   0.267  0.5076511   0.000000000
    ##   0.268  0.5076511   0.000000000
    ##   0.269  0.5076511   0.000000000
    ##   0.270  0.5076511   0.000000000
    ##   0.271  0.5076511   0.000000000
    ##   0.272  0.5076511   0.000000000
    ##   0.273  0.5076511   0.000000000
    ##   0.274  0.5076511   0.000000000
    ##   0.275  0.5076511   0.000000000
    ##   0.276  0.5076511   0.000000000
    ##   0.277  0.5076511   0.000000000
    ##   0.278  0.5076511   0.000000000
    ##   0.279  0.5076511   0.000000000
    ##   0.280  0.5076511   0.000000000
    ##   0.281  0.5076511   0.000000000
    ##   0.282  0.5076511   0.000000000
    ##   0.283  0.5076511   0.000000000
    ##   0.284  0.5076511   0.000000000
    ##   0.285  0.5076511   0.000000000
    ##   0.286  0.5076511   0.000000000
    ##   0.287  0.5076511   0.000000000
    ##   0.288  0.5076511   0.000000000
    ##   0.289  0.5076511   0.000000000
    ##   0.290  0.5076511   0.000000000
    ##   0.291  0.5076511   0.000000000
    ##   0.292  0.5076511   0.000000000
    ##   0.293  0.5076511   0.000000000
    ##   0.294  0.5076511   0.000000000
    ##   0.295  0.5076511   0.000000000
    ##   0.296  0.5076511   0.000000000
    ##   0.297  0.5076511   0.000000000
    ##   0.298  0.5076511   0.000000000
    ##   0.299  0.5076511   0.000000000
    ##   0.300  0.5076511   0.000000000
    ##   0.301  0.5076511   0.000000000
    ##   0.302  0.5076511   0.000000000
    ##   0.303  0.5076511   0.000000000
    ##   0.304  0.5076511   0.000000000
    ##   0.305  0.5076511   0.000000000
    ##   0.306  0.5076511   0.000000000
    ##   0.307  0.5076511   0.000000000
    ##   0.308  0.5076511   0.000000000
    ##   0.309  0.5076511   0.000000000
    ##   0.310  0.5076511   0.000000000
    ##   0.311  0.5076511   0.000000000
    ##   0.312  0.5076511   0.000000000
    ##   0.313  0.5076511   0.000000000
    ##   0.314  0.5076511   0.000000000
    ##   0.315  0.5076511   0.000000000
    ##   0.316  0.5076511   0.000000000
    ##   0.317  0.5076511   0.000000000
    ##   0.318  0.5076511   0.000000000
    ##   0.319  0.5076511   0.000000000
    ##   0.320  0.5076511   0.000000000
    ##   0.321  0.5076511   0.000000000
    ##   0.322  0.5076511   0.000000000
    ##   0.323  0.5076511   0.000000000
    ##   0.324  0.5076511   0.000000000
    ##   0.325  0.5076511   0.000000000
    ##   0.326  0.5076511   0.000000000
    ##   0.327  0.5076511   0.000000000
    ##   0.328  0.5076511   0.000000000
    ##   0.329  0.5076511   0.000000000
    ##   0.330  0.5076511   0.000000000
    ##   0.331  0.5076511   0.000000000
    ##   0.332  0.5076511   0.000000000
    ##   0.333  0.5076511   0.000000000
    ##   0.334  0.5076511   0.000000000
    ##   0.335  0.5076511   0.000000000
    ##   0.336  0.5076511   0.000000000
    ##   0.337  0.5076511   0.000000000
    ##   0.338  0.5076511   0.000000000
    ##   0.339  0.5076511   0.000000000
    ##   0.340  0.5076511   0.000000000
    ##   0.341  0.5076511   0.000000000
    ##   0.342  0.5076511   0.000000000
    ##   0.343  0.5076511   0.000000000
    ##   0.344  0.5076511   0.000000000
    ##   0.345  0.5076511   0.000000000
    ##   0.346  0.5076511   0.000000000
    ##   0.347  0.5076511   0.000000000
    ##   0.348  0.5076511   0.000000000
    ##   0.349  0.5076511   0.000000000
    ##   0.350  0.5076511   0.000000000
    ##   0.351  0.5076511   0.000000000
    ##   0.352  0.5076511   0.000000000
    ##   0.353  0.5076511   0.000000000
    ##   0.354  0.5076511   0.000000000
    ##   0.355  0.5076511   0.000000000
    ##   0.356  0.5076511   0.000000000
    ##   0.357  0.5076511   0.000000000
    ##   0.358  0.5076511   0.000000000
    ##   0.359  0.5076511   0.000000000
    ##   0.360  0.5076511   0.000000000
    ##   0.361  0.5076511   0.000000000
    ##   0.362  0.5076511   0.000000000
    ##   0.363  0.5076511   0.000000000
    ##   0.364  0.5076511   0.000000000
    ##   0.365  0.5076511   0.000000000
    ##   0.366  0.5076511   0.000000000
    ##   0.367  0.5076511   0.000000000
    ##   0.368  0.5076511   0.000000000
    ##   0.369  0.5076511   0.000000000
    ##   0.370  0.5076511   0.000000000
    ##   0.371  0.5076511   0.000000000
    ##   0.372  0.5076511   0.000000000
    ##   0.373  0.5076511   0.000000000
    ##   0.374  0.5076511   0.000000000
    ##   0.375  0.5076511   0.000000000
    ##   0.376  0.5076511   0.000000000
    ##   0.377  0.5076511   0.000000000
    ##   0.378  0.5076511   0.000000000
    ##   0.379  0.5076511   0.000000000
    ##   0.380  0.5076511   0.000000000
    ##   0.381  0.5076511   0.000000000
    ##   0.382  0.5076511   0.000000000
    ##   0.383  0.5076511   0.000000000
    ##   0.384  0.5076511   0.000000000
    ##   0.385  0.5076511   0.000000000
    ##   0.386  0.5076511   0.000000000
    ##   0.387  0.5076511   0.000000000
    ##   0.388  0.5076511   0.000000000
    ##   0.389  0.5076511   0.000000000
    ##   0.390  0.5076511   0.000000000
    ##   0.391  0.5076511   0.000000000
    ##   0.392  0.5076511   0.000000000
    ##   0.393  0.5076511   0.000000000
    ##   0.394  0.5076511   0.000000000
    ##   0.395  0.5076511   0.000000000
    ##   0.396  0.5076511   0.000000000
    ##   0.397  0.5076511   0.000000000
    ##   0.398  0.5076511   0.000000000
    ##   0.399  0.5076511   0.000000000
    ##   0.400  0.5076511   0.000000000
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.012.

``` r
train.cart$results
```

    ##        cp  Accuracy        Kappa   AccuracySD     KappaSD
    ## 1   0.000 0.5160627  0.031422285 0.0177530888 0.035774719
    ## 2   0.001 0.5202637  0.038987611 0.0359796788 0.071449567
    ## 3   0.002 0.5370992  0.074622173 0.0227578879 0.044779735
    ## 4   0.003 0.5462800  0.091617630 0.0202513664 0.039816841
    ## 5   0.004 0.5497213  0.097198524 0.0233216505 0.045738186
    ## 6   0.005 0.5518264  0.101056783 0.0187011356 0.036830412
    ## 7   0.006 0.5527824  0.103095230 0.0191989653 0.037933367
    ## 8   0.007 0.5539296  0.105357109 0.0194652414 0.038331227
    ## 9   0.008 0.5541219  0.105790945 0.0199093739 0.039167196
    ## 10  0.009 0.5543135  0.106260417 0.0189049967 0.037220963
    ## 11  0.010 0.5548871  0.107468780 0.0193789085 0.038225818
    ## 12  0.011 0.5566113  0.110831311 0.0186845018 0.036970353
    ## 13  0.012 0.5566113  0.110831311 0.0186845018 0.036970353
    ## 14  0.013 0.5560395  0.109435324 0.0185371974 0.036440824
    ## 15  0.014 0.5560395  0.109435324 0.0185371974 0.036440824
    ## 16  0.015 0.5556571  0.108498649 0.0188720190 0.037260901
    ## 17  0.016 0.5533626  0.103274565 0.0183504533 0.036680307
    ## 18  0.017 0.5533626  0.103274565 0.0183504533 0.036680307
    ## 19  0.018 0.5516366  0.099425969 0.0192074264 0.038516549
    ## 20  0.019 0.5516366  0.099425969 0.0192074264 0.038516549
    ## 21  0.020 0.5516366  0.099425969 0.0192074264 0.038516549
    ## 22  0.021 0.5516366  0.099425969 0.0192074264 0.038516549
    ## 23  0.022 0.5516366  0.099425969 0.0192074264 0.038516549
    ## 24  0.023 0.5525930  0.100943527 0.0189139827 0.038123277
    ## 25  0.024 0.5497195  0.094779809 0.0258080856 0.052853649
    ## 26  0.025 0.5472338  0.089465612 0.0241928937 0.049619640
    ## 27  0.026 0.5455162  0.085690744 0.0237063532 0.048542473
    ## 28  0.027 0.5441778  0.082770506 0.0224058453 0.045702210
    ## 29  0.028 0.5432218  0.080531180 0.0223926971 0.045643770
    ## 30  0.029 0.5424570  0.078703319 0.0221669808 0.045096071
    ## 31  0.030 0.5424570  0.078703319 0.0221669808 0.045096071
    ## 32  0.031 0.5426482  0.078873708 0.0221439708 0.045084776
    ## 33  0.032 0.5426482  0.078873708 0.0221439708 0.045084776
    ## 34  0.033 0.5426482  0.078873708 0.0221439708 0.045084776
    ## 35  0.034 0.5426482  0.078873708 0.0221439708 0.045084776
    ## 36  0.035 0.5426482  0.078873708 0.0221439708 0.045084776
    ## 37  0.036 0.5426482  0.078873708 0.0221439708 0.045084776
    ## 38  0.037 0.5426482  0.078873708 0.0221439708 0.045084776
    ## 39  0.038 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 40  0.039 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 41  0.040 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 42  0.041 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 43  0.042 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 44  0.043 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 45  0.044 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 46  0.045 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 47  0.046 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 48  0.047 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 49  0.048 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 50  0.049 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 51  0.050 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 52  0.051 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 53  0.052 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 54  0.053 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 55  0.054 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 56  0.055 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 57  0.056 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 58  0.057 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 59  0.058 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 60  0.059 0.5378589  0.068604916 0.0199714477 0.040545172
    ## 61  0.060 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 62  0.061 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 63  0.062 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 64  0.063 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 65  0.064 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 66  0.065 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 67  0.066 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 68  0.067 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 69  0.068 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 70  0.069 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 71  0.070 0.5361348  0.064661902 0.0218682436 0.045332343
    ## 72  0.071 0.5309722  0.053574096 0.0221416206 0.046321372
    ## 73  0.072 0.5263833  0.043381671 0.0214703156 0.045709975
    ## 74  0.073 0.5221768  0.034020245 0.0202707150 0.043826565
    ## 75  0.074 0.5149180  0.017618068 0.0169229866 0.036974707
    ## 76  0.075 0.5082258  0.002770348 0.0104856002 0.021842698
    ## 77  0.076 0.5082258  0.002770348 0.0104856002 0.021842698
    ## 78  0.077 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 79  0.078 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 80  0.079 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 81  0.080 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 82  0.081 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 83  0.082 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 84  0.083 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 85  0.084 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 86  0.085 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 87  0.086 0.5057354 -0.003079654 0.0061040095 0.009738721
    ## 88  0.087 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 89  0.088 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 90  0.089 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 91  0.090 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 92  0.091 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 93  0.092 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 94  0.093 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 95  0.094 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 96  0.095 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 97  0.096 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 98  0.097 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 99  0.098 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 100 0.099 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 101 0.100 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 102 0.101 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 103 0.102 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 104 0.103 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 105 0.104 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 106 0.105 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 107 0.106 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 108 0.107 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 109 0.108 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 110 0.109 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 111 0.110 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 112 0.111 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 113 0.112 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 114 0.113 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 115 0.114 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 116 0.115 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 117 0.116 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 118 0.117 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 119 0.118 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 120 0.119 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 121 0.120 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 122 0.121 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 123 0.122 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 124 0.123 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 125 0.124 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 126 0.125 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 127 0.126 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 128 0.127 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 129 0.128 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 130 0.129 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 131 0.130 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 132 0.131 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 133 0.132 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 134 0.133 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 135 0.134 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 136 0.135 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 137 0.136 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 138 0.137 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 139 0.138 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 140 0.139 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 141 0.140 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 142 0.141 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 143 0.142 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 144 0.143 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 145 0.144 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 146 0.145 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 147 0.146 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 148 0.147 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 149 0.148 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 150 0.149 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 151 0.150 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 152 0.151 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 153 0.152 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 154 0.153 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 155 0.154 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 156 0.155 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 157 0.156 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 158 0.157 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 159 0.158 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 160 0.159 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 161 0.160 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 162 0.161 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 163 0.162 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 164 0.163 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 165 0.164 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 166 0.165 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 167 0.166 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 168 0.167 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 169 0.168 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 170 0.169 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 171 0.170 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 172 0.171 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 173 0.172 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 174 0.173 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 175 0.174 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 176 0.175 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 177 0.176 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 178 0.177 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 179 0.178 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 180 0.179 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 181 0.180 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 182 0.181 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 183 0.182 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 184 0.183 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 185 0.184 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 186 0.185 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 187 0.186 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 188 0.187 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 189 0.188 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 190 0.189 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 191 0.190 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 192 0.191 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 193 0.192 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 194 0.193 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 195 0.194 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 196 0.195 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 197 0.196 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 198 0.197 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 199 0.198 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 200 0.199 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 201 0.200 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 202 0.201 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 203 0.202 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 204 0.203 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 205 0.204 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 206 0.205 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 207 0.206 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 208 0.207 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 209 0.208 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 210 0.209 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 211 0.210 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 212 0.211 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 213 0.212 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 214 0.213 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 215 0.214 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 216 0.215 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 217 0.216 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 218 0.217 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 219 0.218 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 220 0.219 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 221 0.220 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 222 0.221 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 223 0.222 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 224 0.223 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 225 0.224 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 226 0.225 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 227 0.226 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 228 0.227 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 229 0.228 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 230 0.229 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 231 0.230 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 232 0.231 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 233 0.232 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 234 0.233 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 235 0.234 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 236 0.235 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 237 0.236 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 238 0.237 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 239 0.238 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 240 0.239 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 241 0.240 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 242 0.241 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 243 0.242 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 244 0.243 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 245 0.244 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 246 0.245 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 247 0.246 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 248 0.247 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 249 0.248 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 250 0.249 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 251 0.250 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 252 0.251 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 253 0.252 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 254 0.253 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 255 0.254 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 256 0.255 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 257 0.256 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 258 0.257 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 259 0.258 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 260 0.259 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 261 0.260 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 262 0.261 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 263 0.262 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 264 0.263 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 265 0.264 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 266 0.265 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 267 0.266 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 268 0.267 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 269 0.268 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 270 0.269 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 271 0.270 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 272 0.271 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 273 0.272 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 274 0.273 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 275 0.274 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 276 0.275 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 277 0.276 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 278 0.277 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 279 0.278 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 280 0.279 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 281 0.280 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 282 0.281 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 283 0.282 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 284 0.283 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 285 0.284 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 286 0.285 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 287 0.286 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 288 0.287 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 289 0.288 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 290 0.289 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 291 0.290 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 292 0.291 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 293 0.292 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 294 0.293 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 295 0.294 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 296 0.295 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 297 0.296 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 298 0.297 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 299 0.298 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 300 0.299 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 301 0.300 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 302 0.301 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 303 0.302 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 304 0.303 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 305 0.304 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 306 0.305 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 307 0.306 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 308 0.307 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 309 0.308 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 310 0.309 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 311 0.310 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 312 0.311 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 313 0.312 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 314 0.313 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 315 0.314 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 316 0.315 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 317 0.316 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 318 0.317 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 319 0.318 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 320 0.319 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 321 0.320 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 322 0.321 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 323 0.322 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 324 0.323 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 325 0.324 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 326 0.325 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 327 0.326 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 328 0.327 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 329 0.328 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 330 0.329 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 331 0.330 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 332 0.331 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 333 0.332 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 334 0.333 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 335 0.334 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 336 0.335 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 337 0.336 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 338 0.337 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 339 0.338 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 340 0.339 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 341 0.340 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 342 0.341 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 343 0.342 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 344 0.343 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 345 0.344 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 346 0.345 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 347 0.346 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 348 0.347 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 349 0.348 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 350 0.349 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 351 0.350 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 352 0.351 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 353 0.352 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 354 0.353 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 355 0.354 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 356 0.355 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 357 0.356 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 358 0.357 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 359 0.358 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 360 0.359 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 361 0.360 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 362 0.361 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 363 0.362 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 364 0.363 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 365 0.364 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 366 0.365 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 367 0.366 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 368 0.367 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 369 0.368 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 370 0.369 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 371 0.370 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 372 0.371 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 373 0.372 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 374 0.373 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 375 0.374 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 376 0.375 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 377 0.376 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 378 0.377 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 379 0.378 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 380 0.379 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 381 0.380 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 382 0.381 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 383 0.382 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 384 0.383 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 385 0.384 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 386 0.385 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 387 0.386 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 388 0.387 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 389 0.388 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 390 0.389 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 391 0.390 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 392 0.391 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 393 0.392 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 394 0.393 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 395 0.394 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 396 0.395 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 397 0.396 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 398 0.397 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 399 0.398 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 400 0.399 0.5076511  0.000000000 0.0007806443 0.000000000
    ## 401 0.400 0.5076511  0.000000000 0.0007806443 0.000000000

``` r
ggplot(train.cart$results, aes(x = cp, y = Accuracy)) + geom_point(size = 2) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))
```

![](SentimentAnalysis_files/figure-markdown_github/unnamed-chunk-25-1.png)

Let us compute the accuracy for the cross-validated CART model:

``` r
mod_cart = train.cart$finalModel

predict_cart = predict(mod_cart, newdata = sofTest, type = "class") 
table(sofTest$useful, predict_cart)
```

    ##    predict_cart
    ##       0   1
    ##   0 730 407
    ##   1 546 557

``` r
acc_cart <- tableAccuracy(sofTest$useful, predict_cart)
acc_cart
```

    ## [1] 0.5745536

``` r
tpr_cart <- tableTPR(sofTest$useful, predict_cart)
tpr_cart
```

    ## [1] 0.5049864

``` r
fpr_cart <- tableFPR(sofTest$useful, predict_cart)
fpr_cart
```

    ## [1] 0.3579595

The accuracy of cross- validated CART model is 0.574, slightly better than the default Logistic Regression. The TPR is identical as of logistic regression while the FPR is slightly higher.

### b3: Default Random Forest

The accuracies of Logistic Regression and cross-validated CART model do not appear to be ideal. How would ensemble methods such as the default RF work?

``` r
set.seed(311)
mod_rf = randomForest(useful ~ ., data = sofTrain)
```

``` r
predict_rf = predict(mod_rf, newdata = sofTest)
```

    ## Error in predict(mod_rf, newdata = sofTest): object 'mod_rf' not found

``` r
table(sofTest$useful, predict_rf)
```

    ## Error in table(sofTest$useful, predict_rf): object 'predict_rf' not found

``` r
acc_rf <- tableAccuracy(sofTest$useful, predict_rf)
```

    ## Error in table(test, pred): object 'predict_rf' not found

``` r
acc_rf
```

    ## Error in eval(expr, envir, enclos): object 'acc_rf' not found

``` r
tpr_rf <- tableTPR(sofTest$useful, predict_rf)
```

    ## Error in table(test, pred): object 'predict_rf' not found

``` r
tpr_rf
```

    ## Error in eval(expr, envir, enclos): object 'tpr_rf' not found

``` r
fpr_rf <- tableFPR(sofTest$useful, predict_rf)
```

    ## Error in table(test, pred): object 'predict_rf' not found

``` r
fpr_rf
```

    ## Error in eval(expr, envir, enclos): object 'fpr_rf' not found

Suprisingly, the default RF does not perform as well as a single CART on the test dataset, with a lower accuracy, lower TPR and higher FPR.

### b4: Cross Validated RF:

Values of possible *mtry* ranges from 1 to p, where p = 120 is the number of features in this problem. The theoretical ideal value is approximately $\\sqrt {p} = 10.95$, so I will be cross-validating from *mtry* = 5 to *mtry* = 15. To select the best value of *mtry*, I attempted 5-fold, 7-fold, and 10-fold cross validation because I think the values provide good compromise between bias, variance and computation time. After experimentation, I choose 5-fold cross validation because it is the least computationally expensive. Performing Cross Validation on an ensembled method like *Random Forest* is much more computationally demanding than performing CV on a single CART, so I opt for 5-fold CV instead of 10-fold CV in this case.

``` r
set.seed(311)
train.rf = train(useful ~ .,
                 data = sofTrain,
                 method = "rf",
                 tuneGrid = data.frame(mtry = 5:15),
                 trControl = trainControl(method = "cv", number = 5))
```

``` r
train.rf
```

    ## Error in eval(expr, envir, enclos): object 'train.rf' not found

``` r
train.rf$results
```

    ## Error in eval(expr, envir, enclos): object 'train.rf' not found

``` r
ggplot(train.rf$results, aes(x = mtry, y = Accuracy)) + geom_point(size = 2) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))
```

    ## Error in ggplot(train.rf$results, aes(x = mtry, y = Accuracy)): object 'train.rf' not found

Based on the model with best mtry = 5, we predict the test set:

``` r
mod_rf2 = train.rf$finalModel
predict_rf2 = predict(mod_rf2, newdata = sofTest)
table(sofTest$useful, predict_rf2)
acc_rf2 <- tableAccuracy(sofTest$useful, predict_rf2)
acc_rf2

tpr_rf2 <- tableTPR(sofTest$useful, predict_rf2)
tpr_rf2

fpr_rf2 <- tableFPR(sofTest$useful, predict_rf2)
fpr_rf2
```

As we can observe from the plot, although *mtry* = 5 returns us the highest accuracy, it is likely not the best model because of the trend the plot displays when mtry is smaller than 5. I wish I could try with *mtry* = 1:20. However, given the time constraint retraining the rf network will not be accompllished.

### b5: Boosting

The final model I will try to evaluate against is boosting model. Unfortunately, due to my limited computation resources, it is impractical for me to conduct cross validation across all of the parameters. Thus, I decide to only conduct 5-fold cross validation on *interaction depth*. Selection of *n.trees* is inspired from lab and previous assignments, while value of *shrinkage* is from the holistic evaluation from gbm documentation and my computing resource. *n.minobsinnode* = 10 is the default value. Again these parameters may not indicate the best performances. For highest accuracy we should cross validate across more parameters.

We will build the model again with 5-fold CV:

``` r
tGrid = expand.grid(n.trees = 1500, interaction.depth = c(8,10,12),
                    shrinkage = 0.1, n.minobsinnode = 10)

set.seed(232)
train.boost <- train(useful ~ .,
                     data = sofTrain,
                     method = "gbm",
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number= 5),
                     verbose = FALSE,
                     metric = "Accuracy",
                     distribution = "bernoulli")
```

``` r
train.boost
```

    ## Error in eval(expr, envir, enclos): object 'train.boost' not found

``` r
train.boost$results
```

    ## Error in eval(expr, envir, enclos): object 'train.boost' not found

``` r
ggplot(train.boost$results, aes(x = interaction.depth, y = Accuracy)) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))
```

    ## Error in ggplot(train.boost$results, aes(x = interaction.depth, y = Accuracy)): object 'train.boost' not found

Based on the model with best interaction.depth = 12 , we predict the test set:

``` r
library(caret)
mod_boost = train.boost$finalModel
```

    ## Error in eval(expr, envir, enclos): object 'train.boost' not found

``` r
sofTest_mm = as.data.frame(model.matrix(useful ~ . +0, data = sofTest))
predict_boost = predict(mod_boost, newdata = sofTest_mm, n.trees = 1500, type = "response")
```

    ## Error in predict(mod_boost, newdata = sofTest_mm, n.trees = 1500, type = "response"): object 'mod_boost' not found

``` r
table(sofTest$useful, predict_boost < 0.5) # for some reason the probabilities are flipped in gbm
```

    ## Error in table(sofTest$useful, predict_boost < 0.5): object 'predict_boost' not found

``` r
acc_boost <- tableAccuracy(sofTest$useful, predict_boost < 0.5)
```

    ## Error in table(test, pred): object 'predict_boost' not found

``` r
acc_boost
```

    ## Error in eval(expr, envir, enclos): object 'acc_boost' not found

``` r
tpr_boost <- tableTPR(sofTest$useful, predict_boost < 0.5)
```

    ## Error in table(test, pred): object 'predict_boost' not found

``` r
tpr_boost
```

    ## Error in eval(expr, envir, enclos): object 'tpr_boost' not found

``` r
fpr_boost <- tableFPR(sofTest$useful, predict_boost < 0.5)
```

    ## Error in table(test, pred): object 'predict_boost' not found

``` r
fpr_boost
```

    ## Error in eval(expr, envir, enclos): object 'fpr_boost' not found

Unfortunately, the boosting tree does not perform as my expectation. The reason is that the parameters are off. From the plot, we can observe that accuracy increases both when the interaction\_depth &lt; 8 and interaction\_depth &gt; 12. This suggests that the best interaction\_depth is not in the range of \[8, 12\]. The number of iterations (n.trees) is also lower than my original intention of n.trees = 3000 (or better find out through CV). The value of learning rate should be lower than 0.1 (smaller steps) to get better accuracies. Due to the time constraint, this is the best work I can present before deadline.

### b6: Final model & Bootstrap:

Let us summarize our findings:

``` r
cat("Accuracy of basic log model: ", acc_log)
```

    ## Accuracy of basic log model:  0.56875

``` r
cat("\nTrue positive rate of basic log model: ", tpr_log)
```

    ## 
    ## True positive rate of basic log model:  0.5049864

``` r
cat("\nFalse positive rate of basic log model: ", fpr_log)
```

    ## 
    ## False positive rate of basic log model:  0.3693931

``` r
cat("Accuracy of CART model: ", acc_cart)
```

    ## Accuracy of CART model:  0.5745536

``` r
cat("\nTrue positive rate of CART model: ", tpr_cart)
```

    ## 
    ## True positive rate of CART model:  0.5049864

``` r
cat("\nFalse positive rate of CART model: ", fpr_cart)
```

    ## 
    ## False positive rate of CART model:  0.3579595

``` r
cat("\nAccuracy of default RF model: ", acc_rf)
```

    ## Error in cat("\nAccuracy of default RF model: ", acc_rf): object 'acc_rf' not found

``` r
cat("\nTrue positive rate of default RF model: ", tpr_rf)
```

    ## Error in cat("\nTrue positive rate of default RF model: ", tpr_rf): object 'tpr_rf' not found

``` r
cat("\nFalse positive rate of default RF model: ", fpr_rf)
```

    ## Error in cat("\nFalse positive rate of default RF model: ", fpr_rf): object 'fpr_rf' not found

``` r
cat("\nAccuracy of cross validated RF model: ", acc_rf2)
```

    ## Error in cat("\nAccuracy of cross validated RF model: ", acc_rf2): object 'acc_rf2' not found

``` r
cat("\nTrue positive rate of cross validated RF model: ", tpr_rf2)
```

    ## Error in cat("\nTrue positive rate of cross validated RF model: ", tpr_rf2): object 'tpr_rf2' not found

``` r
cat("\nFalse positive rate of cross validated RF model: ", fpr_rf2)
```

    ## Error in cat("\nFalse positive rate of cross validated RF model: ", fpr_rf2): object 'fpr_rf2' not found

``` r
cat("\nAccuracy of boosting model: ", acc_boost)
```

    ## Error in cat("\nAccuracy of boosting model: ", acc_boost): object 'acc_boost' not found

``` r
cat("\nTrue positive rate of boosting model: ", tpr_boost)
```

    ## Error in cat("\nTrue positive rate of boosting model: ", tpr_boost): object 'tpr_boost' not found

``` r
cat("\nFalse positive rate of boosting model: ", fpr_boost)
```

    ## Error in cat("\nFalse positive rate of boosting model: ", fpr_boost): object 'fpr_boost' not found

From the models I have built, I think I will employ CART as my final model. This is because it has the highest accuracy and the lowest FPR. However, I think a RF with better selection of mtry values and a boosting model with smaller learning rate, higher number of iterations and better interaction depth will outperform the CART model. Furthermore, it would be great if there was the opportunity to expore stepwise log regression.

Now let us assess the performance of our final CART model through bootstrap:

``` r
all_metrics <- function(test, pred) {
acc <- tableAccuracy(test, pred)

tpr <- tableTPR(test, pred)
fpr <- tableFPR(test, pred)

return(acc, tpr, fpr)
}
```

``` r
set.seed(7191)
CART_test_set = data.frame(response = sofTest$useful, prediction = predict_cart, baseline = mean(sofTest$useful))
CART_boot <- boot(CART_test_set, all_metrics, R = 10000)
CART_boot
```

c:
--

i: Since our model is only recommanding posts that the model predicts useful, we should ensure that out of all the posts the model predicts useful, there is maximum chance that the predicted useful posts are actually useful. In other words, I think we should select the model with the **highest precision**.

In comparison, the baseline stack overflow model predicts every question as useful and recommends them indifferently.

ii: Let us create a function and compute precision:

``` r
tablePREC <- function(test, pred){
  t = table(test, pred)
  prec = t[4]/(t[4] + t[3])
  return (prec)
}
```

And let us compute the precisions of all models:

``` r
prec_log = tablePREC(sofTest$useful, predict_log > 0.5)
prec_cart = tablePREC(sofTest$useful, predict_cart)
prec_rf = tablePREC(sofTest$useful, predict_rf)
```

    ## Error in table(test, pred): object 'predict_rf' not found

``` r
prec_rf2 = tablePREC(sofTest$useful, predict_rf2)
```

    ## Error in table(test, pred): object 'predict_rf2' not found

``` r
prec_boost = tablePREC(sofTest$useful, predict_boost < 0.5)
```

    ## Error in table(test, pred): object 'predict_boost' not found

``` r
cat("Precision of log model is: ", prec_log)
```

    ## Precision of log model is:  0.5701126

``` r
cat("\nPrecision of cart model is: ", prec_cart)
```

    ## 
    ## Precision of cart model is:  0.5778008

``` r
cat("\nPrecision of default rf model is: ", prec_rf)
```

    ## Error in cat("\nPrecision of default rf model is: ", prec_rf): object 'prec_rf' not found

``` r
cat("\nPrecision of CV rf model is: ", prec_rf2)
```

    ## Error in cat("\nPrecision of CV rf model is: ", prec_rf2): object 'prec_rf2' not found

``` r
cat("\nPrecision of boosting model is: ", prec_boost)
```

    ## Error in cat("\nPrecision of boosting model is: ", prec_boost): object 'prec_boost' not found

Again the **CART Model** from part b) returns the highest precision.

Since the baseline model always predicts the questions as useful because it recommends every question submitted, we can get its precision from:

``` r
table(sofTest$useful)
```

    ## 
    ##    0    1 
    ## 1137 1103

``` r
prec_base = 1103/(1103+1137)
cat("Precision of baseline is: ", prec_base)
```

    ## Precision of baseline is:  0.4924107

I estimate the **improved probability** as the difference between 2 precisions:

``` r
cat("Estimate of improved probability: ", prec_cart - prec_base)
```

    ## Estimate of improved probability:  0.08539012
