Author: Ran Meng

This repo contains code of implementing different ML algorithms for predictive modelling:  

*VehicleSalesPredictions*:  This problem involves predicting monthly sales of Jeep Wrangler with data available from Jan 2010- Dec 2018. Features include Unemployment Rate, Queries (Normalized approximation of the number of Google Searches) for "Jeep Wrangler ", Consumer Price Index, and sales of competing vehicles. **Linear Regression** models of different selection of features are applied to fit the data. During the process of making the predictions, I examined the significance of variables as well as the correlations between them with *p-values* and *VIFs*. 


*Letter Recognition*: This problem involves a multi-class classification of letters using **Logistic Regression, decision trees(CART) with pruning and cross-validation, and a cross-validated Random Forests model**. Outcomes of different models are compared and discussed through metrics such as *TPR*, *FPR* and the *ROC* curve.

*Sentiment Analysis*: This problem involves analyzing content of StackOverFlow questions and classify whether the questions are "useful" or not and making the decisions on whether the questions are worth promoting to the top page of SOF. Data is unstructured text and NLP techniques are applied to clean the text and perform feature engineering for machine learning modelling.  **Cross- validated CART and Random forests, Logistic Regressions, as well as cross-validated boosting models** are trained and tested. **Bootstrap** are used for model validations.  

*Songs Recommendations*: This problem involves constructing a training set matrix with imputation based on observed training data by using the softImpute package from R. The performance of **Low- rank Collaborate Filtering Model, regression-based models, RF and CART models** are compared via metrics including *MAE* and *RMSE*. 
