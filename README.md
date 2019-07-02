# Modeling Credit Risk with Decision Trees
The purpose of this project is to build a Decision Tree model that can predict credit worthiness. The model will be trained on the German Credit dataset available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). The dataset contains 1,000 records and 62 features. One of the features describes the applicant's credit risk as either "Good" or "Bad". The other features are attributes related to the account or applicant, such as credit history, loan purpose, and employment duration. Because there is a surplus of attributes, feature selection will be performed to find the most optimal set of predictors.

Decision Trees are algorithms that split features into branching paths to arrive at a prediction. Decision Trees are suited to datasets with many features and may perform better than Linear or Logistic Regression models. 

The results of this analysis will be presented.
* Building a Decision Tree Model with Hyperparameter Tuning
* Visualization of the Decision Tree Model
* Feature Importance with Random Forest
* Advanced Decision Trees: Comparing Boosting-Based Algorithms

The code used to generate the models can be found in the repo's [Jupyter Notebook](https://github.com/jushih/Credit-Risk-Prediction/blob/master/Predicting_Credit_Risk_with_Decision_Trees.ipynb).

## Decision Tree Model

I begin by building a simple decision tree model. To find the best-fit decision tree parameters (e.g., number of features and depth), I perform hyperparameter tuning with grid search and cross-validation. Cross-validation is a method of holding out a subset of the data from training (in this case, 30%) and using this test set to validate the model's prediction. Grid-search is the process of testing different hyperparameter combinations to find the optimal parameters for the model.

The Grid Search found the following best parameters. 

```{'max_depth': 3, 'max_features': 8}```

From the best-fit model, I construct a confusion matrix and calculate model accuracy.

![dtreematrix](https://github.com/jushih/Credit-Risk-Prediction/blob/master/decision_tree_matrix.png)

* Accuracy on test data: 0.727

I use dtreeviz to visualize my Decision Tree model with max_depth: 3 and max_features: 8. From this visualization, we can see the cutoff point at each feature level that goes into determining credit risk. We can also see where it misclassifies the applicant. 

![dtreeviz](https://github.com/jushih/Credit-Risk-Prediction/blob/master/decision_tree_viz.png "Visualization of Decision Tree Model")


## Random Forest Model

### Improving Accuracy with Random Forest

Next, I'll try to improve classification accuracy by building a Random Forest ensemble model. Ensemble techniques are where weak learners combine their predictions by averaging or taking a max vote to create a strong learner. Random Forests are comprised of multiple decision trees that take a random sample of the features to form their prediction, and then decide the final classification by consensus vote from all the trees.

The advantage of a Random Forest model over a simple Decision Tree is that Decision Trees are prone to overfitting. Decision Trees, especially ones that are deep, will form detailed feature branches that fit the training data but don't generalize well. In the case of a Random Forest model, each tree uses a subsample of the features to make their prediction so overfitting is less likely to occur. This sampling technique is known as bootstrap aggregation, or **bagging**. It leads to better model accuracy and also allows Random Forests another advantage - the ability to determine **feature importance**.

I also perform hyperparameter tuning on the Random Forest to build the best model. The Grid Search found the following best parameters.

```
{'bootstrap': True,
 'max_depth': None,
 'max_features': 'auto',
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 200}
 ```

![rfmatrix](https://github.com/jushih/Credit-Risk-Prediction/blob/master/random_forest_matrix.png)

* Accuracy on test data: 0.760

By using an ensemble method, the accuracy has improved 4% compared to the simple Decision Tree model.


### Feature Importance with Random Forest

![features](https://github.com/jushih/Credit-Risk-Prediction/blob/master/feature_importance.png)

The most predictive features in modeling credit risk is Amount, Duration, Age, and Checking Account Status - None. The features selected by the simple Decision Tree model are among the top ten important features, giving confidence to the model.

## Advanced Decision Trees: Comparing Boosting-Based Algorithms

Like bagging, **boosting** is another ensemble technique that incorporates more than one decision tree. After the first decision tree is trained, boosting algorithms perform subsequent training by placing weight on data that is hard to classify and less weight on data that is easy to classify. It uses a loss function to measure error and correct for it in the next iteration. Boosted-tree algorithms also penalize models for complexity. The prediction of the final boosted-tree model is the weighted sum of the predictions made by the individual models. Below are three gradient-boosted trees that I will use to predict credit risk and compare performance:

* **XGBoost** - A popular implementation of gradient boosted trees designed for speed and performance. At the time of its publication, it was unmatched and won many Kaggle competitions. A disadvantage of XGBoost is that it requires data to be stored in memory when run. Since then, newer algorithms have been developed that do not have to process data in-memory.
* **Catboost** - The newest of the three algorithms, CatBoost is designed to handle categorical features better. Instead of one-hot encoding features, which causes the curse of dimensionality, CatBoost transforms categorical features into values based on a stastistical calculation of its relationship with the target variable. CatBoost also divides a given dataset into random permutations and applies ordered boosting on those random permutations.
* **LightGBM** - Designed to be a faster implementation of XGBoost with similar accuracy. This algorithm inspects the most informative samples and skips non-informative samples. It also bins sparse features, reducing complexity.

After training the models using grid search and cross-validation, the accuracy scores of the models are as follows.

<img src="https://github.com/jushih/Credit-Risk-Prediction/blob/master/model_comparison.png" alt="modelcomp" width="350"/>

There was a jump in accuracy from the Decision Tree model to the ensemble models, which makes sense since ensemble models are designed to perform better through consensus prediction. The CatBoost model outperformed the other models with the highest accuracy of 78.7%. This dataset contained many categorical features suited to using the CatBoost algorithm.
