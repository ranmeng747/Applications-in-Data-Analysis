VehicleSalesPredictions
================
Ran Meng
February 10, 2019

Set directory:

``` r
setwd("C:/Ran/Berkeley/IEOR/242/Applications in Data Analysis")
library(dplyr)
library(ggplot2)
library(GGally)
library(car)
```

Split the data into training set (2010- 2015) and test set (2016 - 2018) and train with the 4 independent variables:

``` r
# Load data:
jeep <- read.csv("Wrangler242-Spring2019.csv") 

# split into training and test sets 

jeep.train <- filter(jeep, Year <= 2015) 
jeep.test <- filter(jeep, Year > 2015)

# train the model
#lm(y~x1+x2+...,data)
mod <- lm(WranglerSales ~ Unemployment + WranglerQueries + CPI.Energy + CPI.All,
           data = jeep.train)
summary(mod)
```

    ## 
    ## Call:
    ## lm(formula = WranglerSales ~ Unemployment + WranglerQueries + 
    ##     CPI.Energy + CPI.All, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3453.6 -1178.0   -51.9   887.1  7628.9 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     -51248.45   59358.29  -0.863    0.391    
    ## Unemployment       989.12    1122.98   0.881    0.382    
    ## WranglerQueries    274.55      34.25   8.015 2.26e-11 ***
    ## CPI.Energy          -8.64      31.65  -0.273    0.786    
    ## CPI.All            192.02     253.11   0.759    0.451    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1757 on 67 degrees of freedom
    ## Multiple R-squared:  0.7961, Adjusted R-squared:  0.7839 
    ## F-statistic:  65.4 on 4 and 67 DF,  p-value: < 2.2e-16

``` r
vif(mod)
```

    ##    Unemployment WranglerQueries      CPI.Energy         CPI.All 
    ##       68.852495        4.509533        8.410345       69.700881

a:
--

Removing the varaible with the highest VIF (CPI.All) and retrain the model:

``` r
# train the new model
mod_a <- lm(WranglerSales ~ Unemployment + WranglerQueries + CPI.Energy,
           data = jeep.train)
summary(mod_a)
```

    ## 
    ## Call:
    ## lm(formula = WranglerSales ~ Unemployment + WranglerQueries + 
    ##     CPI.Energy, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3514.2 -1152.3   -16.9   996.4  7586.1 
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     -6332.67    4258.99  -1.487    0.142    
    ## Unemployment      165.17     284.63   0.580    0.564    
    ## WranglerQueries   279.39      33.55   8.328 5.56e-12 ***
    ## CPI.Energy         13.83      11.14   1.241    0.219    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1751 on 68 degrees of freedom
    ## Multiple R-squared:  0.7944, Adjusted R-squared:  0.7853 
    ## F-statistic: 87.56 on 3 and 68 DF,  p-value: < 2.2e-16

Inspect VIF:

``` r
vif(mod_a)
```

    ##    Unemployment WranglerQueries      CPI.Energy 
    ##        4.450958        4.353009        1.048267

Remove "Unemployment" because it seems correlating with "WranglerQueries" and is insignificant:

``` r
# train the new model
mod_a <- lm(WranglerSales ~ WranglerQueries + CPI.Energy,
           data = jeep.train)
summary(mod_a)
```

    ## 
    ## Call:
    ## lm(formula = WranglerSales ~ WranglerQueries + CPI.Energy, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3585.1 -1120.9     6.2   883.0  7576.0 
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     -4463.50    2772.88  -1.610    0.112    
    ## WranglerQueries   262.34      16.14  16.259   <2e-16 ***
    ## CPI.Energy         14.95      10.92   1.369    0.175    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1743 on 69 degrees of freedom
    ## Multiple R-squared:  0.7933, Adjusted R-squared:  0.7873 
    ## F-statistic: 132.4 on 2 and 69 DF,  p-value: < 2.2e-16

Inspect VIF:

``` r
vif(mod_a)
```

    ## WranglerQueries      CPI.Energy 
    ##        1.016816        1.016816

#### i :

Linear equation produced by the new model: \[

y = B\_0 + B\_1WranglerQueries + B\_2CPI.Energy

\]

The interpretation of coefficients are:

-   B<sub>1</sub> = 262.34(An additional increase in 1 unit of WranglerQueries is expected to result in gaining 262.34 units of sale)
-   B<sub>2</sub> = 14.95 (An additional increase in 1 unit of CPI.Energy is expected to result in gaining 14.95 units of sale)

#### ii :

By Observing the VIF scores of the original model, we can see that 'CPI.ALL' has the highest VIF score (69.7). Thus, I removed the variable to reduce the multicollinearity and hence the model is less susceptible to noise in the training data. After rerunning the model, we observe that the VIF scores for all 3 features are &lt; 5. I tried to further mitigate the issue of multicollinearity by removing the variable "Unemployment" because it apperaed corrlating with WranglerQueries and it was a less significant variable.

#### iii:

By observing the p-values, we can conclude that "WranglerQueries" is ths most significant variable with the smallest p-value of &lt; 2e-16 indicated by \*\*\*.

#### iv:

I think the signs of the coefficients make sense. For the positive "WranglerQueries"" coefficient, it is saying that there will be more sales of Wrangler vehicles if people researched about the vehicle more, and I consider the number of queries as an indication of the level of the interest. In other words, if people are more interested, they will search the term "Jeep Wranglers" more and increase the number of Wrangler sales. For the CPI.Energy, it is important to realize that the feature does not have a significant impact on the sales from its p-value, and its coefficient is fairly small. CPI covers an extensive range of items and does not necessarily represent the price- change experience of individuals.

#### v:

Summary of my model with "WranglerQueries" and "CPI.Energy" as features:

``` r
summary(mod_a)
```

    ## 
    ## Call:
    ## lm(formula = WranglerSales ~ WranglerQueries + CPI.Energy, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3585.1 -1120.9     6.2   883.0  7576.0 
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     -4463.50    2772.88  -1.610    0.112    
    ## WranglerQueries   262.34      16.14  16.259   <2e-16 ***
    ## CPI.Energy         14.95      10.92   1.369    0.175    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1743 on 69 degrees of freedom
    ## Multiple R-squared:  0.7933, Adjusted R-squared:  0.7873 
    ## F-statistic: 132.4 on 2 and 69 DF,  p-value: < 2.2e-16

Final equation of the new model:
*y* = −4463.50 + 262.34 ⋅ *W**r**a**n**g**l**e**r**Q**u**e**r**i**e**s* + 14.95 ⋅ *C**P**I*.*E**n**e**r**g**y*

I think the model fits training data reasonably well as the adjusted R<sup>2</sup> value is close to 80% as the quantifiable metric.

b:
--

Training the model with the addition of "Month Factor":

``` r
# train the model
#lm(y~x1+x2+...,data)
mod_b <- lm(WranglerSales ~ Unemployment + WranglerQueries + CPI.Energy + CPI.All + MonthFactor,
           data = jeep.train)
summary(mod_b)
```

    ## 
    ## Call:
    ## lm(formula = WranglerSales ~ Unemployment + WranglerQueries + 
    ##     CPI.Energy + CPI.All + MonthFactor, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3013.5  -596.8  -144.7   486.1  8019.9 
    ## 
    ## Coefficients:
    ##                       Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          -71969.75   53164.01  -1.354 0.181261    
    ## Unemployment            959.91     993.04   0.967 0.337880    
    ## WranglerQueries         192.08      61.80   3.108 0.002956 ** 
    ## CPI.Energy              -22.78      28.63  -0.796 0.429510    
    ## CPI.All                 318.26     232.50   1.369 0.176508    
    ## MonthFactorAugust      -158.57     935.96  -0.169 0.866076    
    ## MonthFactorDecember     -50.31    1047.63  -0.048 0.961872    
    ## MonthFactorFebruary   -1021.34     896.10  -1.140 0.259240    
    ## MonthFactorJanuary    -3256.91     934.56  -3.485 0.000964 ***
    ## MonthFactorJuly        -303.87     993.73  -0.306 0.760902    
    ## MonthFactorJune          67.03     957.99   0.070 0.944467    
    ## MonthFactorMarch       -158.79     876.76  -0.181 0.856933    
    ## MonthFactorMay         1806.86     902.65   2.002 0.050166 .  
    ## MonthFactorNovember   -1701.16     974.45  -1.746 0.086337 .  
    ## MonthFactorOctober     -731.70     967.74  -0.756 0.452764    
    ## MonthFactorSeptember  -1028.08     878.30  -1.171 0.246745    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1516 on 56 degrees of freedom
    ## Multiple R-squared:  0.873,  Adjusted R-squared:  0.839 
    ## F-statistic: 25.66 on 15 and 56 DF,  p-value: < 2.2e-16

#### i:

The regression equation is:

The interpretation of the coefficients for the four variables "Unemployment", "WranglerSales", "CPI.All" and "CPI.Energy" are identical as those of part a:

-   An additional increase in 1% of unemployment rate is expected to result in gaining 959.91 units of sale)
-   An additional increase in 1 unit of WranglerQueries is expected to result in gaining 192.08 units of sale)
-   An additional increase in 1 unit of CPI.Energy is expected to result in losing 22.78 units of sale)
-   An additional increase in 1 unit of CPI.All is expected to result in gaining 318.26 units of sale)

The coefficients of the 11 MonthFactor Dummy Variables are compared to the last MonthFactor Variable(MonthFactorApril) which has coefficient = 0. If the sign of the coefficients are positive for the respective months, it indicates that Jeep will sell more Wranglers in these months than in April, assuming all other independent variables stay constant. If the sign of the coefficients are negative for the respective months, it indicates that Jeep will sell less Wranglers in these months than in April.

#### ii:

The multiple R<sup>2</sup> for the training set is 0.873 and the adjusted R<sup>2</sup> is 0.839. According to p-values, WranglerQueries and MonthFactorJanuary are the significant variables.

#### iii:

I think adding the variable "MonthFactor" will improve the model. As suggested with the higher value of R<sup>2</sup>, the new model performs better on the training data. Due to the property of linear regression model with high bias but low variance, I predict the OSR<sup>2</sup> for the new model will be higher than that of the original (without MonthFactor), meaning the new model is better at predicting future sales. To quantitatively support my hypothesis, we need to calculate OSR<sup>2</sup> for the respective models.

#### iv:

Yes. For example we can also add "Holiday" as another feature for seasonality because holidays such as Easter Break and Thanksgiving are moving holidays and do not have fixed dates. There could be issues of lead & lag effects of independent variables(no immediate effect on responses) so adding the "Holiday" feature could mitigate the problem of lead & lag.

c:
--

We look at the model from part b with all 5 variables including MonthFactor, and see if there is any variable with high VIF:

``` r
vif(mod_b)
```

    ##                      GVIF Df GVIF^(1/(2*Df))
    ## Unemployment    72.235553  1        8.499150
    ## WranglerQueries 19.693557  1        4.437742
    ## CPI.Energy       9.229663  1        3.038036
    ## CPI.All         78.909535  1        8.883104
    ## MonthFactor      4.665745 11        1.072520

For single coefficient variables(Unemployment, WranglerQueries, CPI.Energy, CPI.All) the VIF equal to the GVIF above. Thus, I decide to remove "CPI.All" as it has the highest VIF/GVIF.

``` r
# train the model
#lm(y~x1+x2+...,data)
mod_c <- lm(WranglerSales ~   Unemployment + CPI.Energy + WranglerQueries + MonthFactor,
           data = jeep.train)
summary(mod_c)
```

    ## 
    ## Call:
    ## lm(formula = WranglerSales ~ Unemployment + CPI.Energy + WranglerQueries + 
    ##     MonthFactor, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3061.5  -688.7   -84.5   537.7  7992.0 
    ## 
    ## Coefficients:
    ##                       Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)            256.834   6559.735   0.039 0.968905    
    ## Unemployment          -247.251    460.022  -0.537 0.593031    
    ## CPI.Energy              14.079      9.789   1.438 0.155853    
    ## WranglerQueries        220.469     58.655   3.759 0.000404 ***
    ## MonthFactorAugust     -187.864    942.863  -0.199 0.842777    
    ## MonthFactorDecember    256.402   1031.202   0.249 0.804530    
    ## MonthFactorFebruary   -983.912    902.526  -1.090 0.280223    
    ## MonthFactorJanuary   -3099.912    934.576  -3.317 0.001588 ** 
    ## MonthFactorJuly       -460.458    994.666  -0.463 0.645179    
    ## MonthFactorJune        -91.368    958.238  -0.095 0.924371    
    ## MonthFactorMarch      -159.170    883.455  -0.180 0.857660    
    ## MonthFactorMay        1726.566    907.616   1.902 0.062187 .  
    ## MonthFactorNovember  -1397.907    956.182  -1.462 0.149242    
    ## MonthFactorOctober    -511.285    961.533  -0.532 0.596973    
    ## MonthFactorSeptember  -954.041    883.330  -1.080 0.284669    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1528 on 57 degrees of freedom
    ## Multiple R-squared:  0.8687, Adjusted R-squared:  0.8365 
    ## F-statistic: 26.94 on 14 and 57 DF,  p-value: < 2.2e-16

Inspect VIF:

``` r
vif(mod_c)
```

    ##                      GVIF Df GVIF^(1/(2*Df))
    ## Unemployment    15.267659  1        3.907385
    ## CPI.Energy       1.063092  1        1.031063
    ## WranglerQueries 17.474691  1        4.180274
    ## MonthFactor      4.121258 11        1.066488

We can observe that "Unemployment" has a high VIF and it is not a significant variable, so remove the variable and retrain the model:

``` r
# train the model
#lm(y~x1+x2+...,data)
mod_c <- lm(WranglerSales ~  CPI.Energy + WranglerQueries + MonthFactor,
           data = jeep.train)
summary(mod_c)
```

    ## 
    ## Call:
    ## lm(formula = WranglerSales ~ CPI.Energy + WranglerQueries + MonthFactor, 
    ##     data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3069.6  -635.9   -10.5   499.0  7952.2 
    ## 
    ## Coefficients:
    ##                       Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          -2987.243   2553.173  -1.170   0.2468    
    ## CPI.Energy              13.141      9.573   1.373   0.1751    
    ## WranglerQueries        250.879     15.369  16.324   <2e-16 ***
    ## MonthFactorAugust     -356.506    883.656  -0.403   0.6881    
    ## MonthFactorDecember    538.814    881.837   0.611   0.5436    
    ## MonthFactorFebruary   -887.206    878.971  -1.009   0.3170    
    ## MonthFactorJanuary   -2945.135    883.637  -3.333   0.0015 ** 
    ## MonthFactorJuly       -695.601    887.792  -0.784   0.4365    
    ## MonthFactorJune       -283.167    883.844  -0.320   0.7498    
    ## MonthFactorMarch      -135.970    876.974  -0.155   0.8773    
    ## MonthFactorMay        1616.812    878.908   1.840   0.0710 .  
    ## MonthFactorNovember  -1202.044    878.581  -1.368   0.1765    
    ## MonthFactorOctober    -307.785    878.415  -0.350   0.7273    
    ## MonthFactorSeptember  -938.728    877.442  -1.070   0.2891    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1519 on 58 degrees of freedom
    ## Multiple R-squared:  0.8681, Adjusted R-squared:  0.8385 
    ## F-statistic: 29.35 on 13 and 58 DF,  p-value: < 2.2e-16

Inspect VIF:

``` r
vif(mod_c)
```

    ##                     GVIF Df GVIF^(1/(2*Df))
    ## CPI.Energy      1.029297  1        1.014543
    ## WranglerQueries 1.214601  1        1.102089
    ## MonthFactor     1.201464 11        1.008378

All variables have satisfactory VIFs by this point. However, I will remove "CPI.Energy" because it appears insignificant:

``` r
# train the model
#lm(y~x1+x2+...,data)
mod_c <- lm(WranglerSales ~  WranglerQueries + MonthFactor,
           data = jeep.train)
summary(mod_c)
```

    ## 
    ## Call:
    ## lm(formula = WranglerSales ~ WranglerQueries + MonthFactor, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2817.4  -610.4   -68.8   561.0  8152.9 
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)            241.81     999.74   0.242   0.8097    
    ## WranglerQueries        247.69      15.31  16.183   <2e-16 ***
    ## MonthFactorAugust     -335.80     890.12  -0.377   0.7073    
    ## MonthFactorDecember    457.85     886.43   0.517   0.6074    
    ## MonthFactorFebruary   -902.89     885.46  -1.020   0.3120    
    ## MonthFactorJanuary   -3003.21     889.21  -3.377   0.0013 ** 
    ## MonthFactorJuly       -690.35     894.41  -0.772   0.4433    
    ## MonthFactorJune       -284.42     890.44  -0.319   0.7505    
    ## MonthFactorMarch      -138.52     883.52  -0.157   0.8759    
    ## MonthFactorMay        1622.89     885.46   1.833   0.0719 .  
    ## MonthFactorNovember  -1246.75     884.53  -1.410   0.1639    
    ## MonthFactorOctober    -330.86     884.81  -0.374   0.7098    
    ## MonthFactorSeptember  -948.84     883.96  -1.073   0.2875    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1530 on 59 degrees of freedom
    ## Multiple R-squared:  0.8638, Adjusted R-squared:  0.8361 
    ## F-statistic: 31.18 on 12 and 59 DF,  p-value: < 2.2e-16

Inspect VIF:

``` r
vif(mod_c)
```

    ##                     GVIF Df GVIF^(1/(2*Df))
    ## WranglerQueries 1.186895  1        1.089447
    ## MonthFactor     1.186895 11        1.007819

Both variables are significant and have low VIFs. This is a good point to finalize the model and evaluate its performance by Computing OSR<sup>2</sup>:

``` r
# compute OSR^2

jeepPredictions <- predict(mod_c, newdata=jeep.test)
# this builds a vector of predicted values on the test set
SSE = sum((jeep.test$WranglerSales - jeepPredictions)^2)
SST = sum((jeep.test$WranglerSales - mean(jeep.train$WranglerSales))^2)
OSR2 = 1 - SSE/SST
cat(OSR2)
```

    ## 0.617641

With the addition of MonthFactor, the R<sup>2</sup> value is 0.864, the adjusted R<sup>2</sup> is 0.836 and the OSR2^2 is 0.618. I am not content with this model as I find the difference between R<sup>2</sup> and OSR2<sup>2</sup> too much and the model is not accurate enough at predicting future sales. It would be interesting to investigate how models with higher flexibility perform with the same dataset.

Equation of my final model for predicting Wrangler Sales:

d:
--

We first examine the model with all independent variables in part(a) and part(b), but with WranglerQueries replaced by GMCQueries, and WranglerSales replaced with GMCSales::

``` r
# train the model
#lm(y~x1+x2+...,data)
mod_d <- lm(GMCSales ~ Unemployment + GMCQueries + CPI.Energy + CPI.All + MonthFactor,
           data = jeep.train)
summary(mod_d)
```

    ## 
    ## Call:
    ## lm(formula = GMCSales ~ Unemployment + GMCQueries + CPI.Energy + 
    ##     CPI.All + MonthFactor, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3922.7  -792.7    38.1   654.9  9469.9 
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          18904.93   73128.48   0.259   0.7970    
    ## Unemployment         -1104.29    1357.71  -0.813   0.4195    
    ## GMCQueries              94.66     103.28   0.917   0.3633    
    ## CPI.Energy              11.80      40.96   0.288   0.7744    
    ## CPI.All                -12.95     325.34  -0.040   0.9684    
    ## MonthFactorAugust     2021.57    1238.14   1.633   0.1081    
    ## MonthFactorDecember   5312.65    1197.20   4.438 4.31e-05 ***
    ## MonthFactorFebruary    203.79    1245.71   0.164   0.8706    
    ## MonthFactorJanuary   -3034.26    1259.14  -2.410   0.0193 *  
    ## MonthFactorJuly        554.23    1245.35   0.445   0.6580    
    ## MonthFactorJune        408.64    1198.42   0.341   0.7344    
    ## MonthFactorMarch        10.49    1238.26   0.008   0.9933    
    ## MonthFactorMay         975.10    1184.44   0.823   0.4139    
    ## MonthFactorNovember     47.75    1192.90   0.040   0.9682    
    ## MonthFactorOctober    1308.78    1186.13   1.103   0.2746    
    ## MonthFactorSeptember   396.74    1189.21   0.334   0.7399    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2050 on 56 degrees of freedom
    ## Multiple R-squared:  0.7848, Adjusted R-squared:  0.7272 
    ## F-statistic: 13.62 on 15 and 56 DF,  p-value: 1.402e-13

Inspecting VIF:

``` r
vif(mod_d)
```

    ##                   GVIF Df GVIF^(1/(2*Df))
    ## Unemployment 73.867989  1        8.594649
    ## GMCQueries   30.988194  1        5.566704
    ## CPI.Energy   10.339682  1        3.215538
    ## CPI.All      84.524656  1        9.193729
    ## MonthFactor   1.597171 11        1.021511

Removing "CPI.All" from the model as it has the highest GVIF:

``` r
# train the model
#lm(y~x1+x2+...,data)
mod_d<- lm(GMCSales ~ Unemployment + GMCQueries + CPI.Energy + MonthFactor,
           data = jeep.train)
summary(mod_d)
```

    ## 
    ## Call:
    ## lm(formula = GMCSales ~ Unemployment + GMCQueries + CPI.Energy + 
    ##     MonthFactor, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3926.1  -795.2    31.5   662.1  9466.6 
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          16023.41   10214.52   1.569   0.1223    
    ## Unemployment         -1060.62     792.57  -1.338   0.1861    
    ## GMCQueries              92.96      93.18   0.998   0.3226    
    ## CPI.Energy              10.25      12.95   0.792   0.4317    
    ## MonthFactorAugust     2022.75    1226.90   1.649   0.1047    
    ## MonthFactorDecember   5313.23    1186.58   4.478 3.67e-05 ***
    ## MonthFactorFebruary    211.96    1217.86   0.174   0.8624    
    ## MonthFactorJanuary   -3027.54    1236.78  -2.448   0.0175 *  
    ## MonthFactorJuly        558.30    1230.23   0.454   0.6517    
    ## MonthFactorJune        410.91    1186.53   0.346   0.7304    
    ## MonthFactorMarch        17.24    1215.81   0.014   0.9887    
    ## MonthFactorMay         974.87    1174.01   0.830   0.4098    
    ## MonthFactorNovember     42.64    1175.54   0.036   0.9712    
    ## MonthFactorOctober    1307.60    1175.32   1.113   0.2706    
    ## MonthFactorSeptember   395.88    1178.55   0.336   0.7382    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2032 on 57 degrees of freedom
    ## Multiple R-squared:  0.7848, Adjusted R-squared:  0.7319 
    ## F-statistic: 14.85 on 14 and 57 DF,  p-value: 3.54e-14

Inspecting VIF:

``` r
vif(mod_d)
```

    ##                   GVIF Df GVIF^(1/(2*Df))
    ## Unemployment 25.620668  1        5.061686
    ## GMCQueries   25.670108  1        5.066568
    ## CPI.Energy    1.051851  1        1.025598
    ## MonthFactor   1.424898 11        1.016226

Removing "GMCQueries"" because of its high VIF and high p-value compared to "Unemployment":

``` r
mod_d<- lm(GMCSales ~ Unemployment + CPI.Energy + MonthFactor,
           data = jeep.train)
summary(mod_d)
```

    ## 
    ## Call:
    ## lm(formula = GMCSales ~ Unemployment + CPI.Energy + MonthFactor, 
    ##     data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3901.1  -862.4   -35.9   602.4  9302.8 
    ## 
    ## Coefficients:
    ##                       Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          25720.297   3141.558   8.187 2.98e-11 ***
    ## Unemployment         -1834.802    161.296 -11.375  < 2e-16 ***
    ## CPI.Energy               9.163     12.904   0.710   0.4805    
    ## MonthFactorAugust     2379.024   1173.730   2.027   0.0473 *  
    ## MonthFactorDecember   5457.830   1177.651   4.635 2.07e-05 ***
    ## MonthFactorFebruary    537.033   1173.418   0.458   0.6489    
    ## MonthFactorJanuary   -2640.767   1174.401  -2.249   0.0284 *  
    ## MonthFactorJuly        925.660   1173.791   0.789   0.4336    
    ## MonthFactorJune        584.524   1173.657   0.498   0.6203    
    ## MonthFactorMarch       334.929   1173.323   0.285   0.7763    
    ## MonthFactorMay        1013.072   1173.334   0.863   0.3915    
    ## MonthFactorNovember     45.097   1175.489   0.038   0.9695    
    ## MonthFactorOctober    1324.848   1175.149   1.127   0.2642    
    ## MonthFactorSeptember   492.761   1174.499   0.420   0.6764    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2032 on 58 degrees of freedom
    ## Multiple R-squared:  0.781,  Adjusted R-squared:  0.732 
    ## F-statistic: 15.91 on 13 and 58 DF,  p-value: 1.373e-14

Inspecting VIF:

``` r
vif(mod_d)
```

    ##                  GVIF Df GVIF^(1/(2*Df))
    ## Unemployment 1.061198  1        1.030145
    ## CPI.Energy   1.044341  1        1.021930
    ## MonthFactor  1.026620 11        1.001195

Removing "CPI.Energy" because of its high p-value/lack of significance:

``` r
mod_d<- lm(GMCSales ~  Unemployment + MonthFactor,
           data = jeep.train)
summary(mod_d)
```

    ## 
    ## Call:
    ## lm(formula = GMCSales ~ Unemployment + MonthFactor, data = jeep.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -4159.8  -841.3   -83.9   619.3  9437.0 
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          27688.52    1472.56  18.803  < 2e-16 ***
    ## Unemployment         -1812.80     157.63 -11.501  < 2e-16 ***
    ## MonthFactorAugust     2381.94    1168.78   2.038   0.0460 *  
    ## MonthFactorDecember   5424.58    1171.77   4.629 2.06e-05 ***
    ## MonthFactorFebruary    532.78    1168.46   0.456   0.6501    
    ## MonthFactorJanuary   -2670.49    1168.71  -2.285   0.0259 *  
    ## MonthFactorJuly        912.99    1168.71   0.781   0.4378    
    ## MonthFactorJune        570.29    1168.55   0.488   0.6273    
    ## MonthFactorMarch       334.64    1168.38   0.286   0.7756    
    ## MonthFactorMay        1009.53    1168.38   0.864   0.3911    
    ## MonthFactorNovember     30.12    1170.35   0.026   0.9796    
    ## MonthFactorOctober    1325.33    1170.20   1.133   0.2620    
    ## MonthFactorSeptember   488.23    1169.54   0.417   0.6779    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2024 on 59 degrees of freedom
    ## Multiple R-squared:  0.7791, Adjusted R-squared:  0.7342 
    ## F-statistic: 17.34 on 12 and 59 DF,  p-value: 4.057e-15

Inspecting VIF:

``` r
vif(mod_d)
```

    ##                  GVIF Df GVIF^(1/(2*Df))
    ## Unemployment 1.022053  1        1.010967
    ## MonthFactor  1.022053 11        1.000992

Computing OSR<sup>2</sup>:

``` r
# compute OSR^2

GMCPredictions <- predict(mod_d, newdata=jeep.test)
# this builds a vector of predicted values on the test set
SSE = sum((jeep.test$GMCSales - GMCPredictions)^2)
SST = sum((jeep.test$GMCSales - mean(jeep.train$GMCSales))^2)
OSR2 = 1 - SSE/SST
cat(OSR2)
```

    ## 0.509097

Equation of my final model for predicting GMC Sierra Sales:

#### e:

Predicting January 2019 sales for Wranglers, we use equation of mod\_c:

Plugging in the values with WranglerQueries = [76](https://trends.google.com/trends/explore?date=2010-01-01%202019-01-31&geo=US&q=Jeep%20Wrangler):

Predicting January 2019 sales for GMC sierra, we use equation of mod\_d:
Plugging in the values with Unemployment = [4.0](http://www.ncsl.org/research/labor-and-employment/national-employment-monthly-update.aspx) for Jan 2019:

I consider the OSR<sup>2</sup> for both the Wrangler and the GMC not ideal since both of them are &lt; 0.7. I am expecting a percentage difference of greater than 10% between my predictions and the actual values. It would be great to see the actual sales for January 2019. However, it is worth noting that a model's performance should not be evaluated just by a single month's observation. Evaluating OSR<sup>2</sup> on large sample of test data is more reliable. In addition, there is no linear assumption on the data so models with higher flexibility may perform better.
