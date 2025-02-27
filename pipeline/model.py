import pandas as pd
import numpy as np
from scipy import stats
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import linear_model
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


class PredictionResults:
    def __init__(self, results, y_train_predicted, y_test_predicted, feature_importance_scores=None, model_hyperparams=None, model_type="RandHypOPt_Ridge_regression"):
        self.results = results
        self.y_train_predicted = y_train_predicted.tolist()
        self.y_test_predicted = y_test_predicted.tolist()
        if not (model_type in ["BayesHypOPt_NN_regression", "RandHypOPt_NN_regression"]):
            self.feature_importance_scores = feature_importance_scores
        else:
            self.feature_importance_scores = feature_importance_scores
        self.model_hyperparams = model_hyperparams


class ModelBuilder:

    def __init__(self, X_train, y_train, X_test, y_test, dataset_name=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.results = None
        self.trained = False
        self.dataset_name = dataset_name

    def train_model(self, train_method):
        if train_method == 'BayesHypOPt_Ridge_regression':
            result = self.BayesHypOPt_Ridge_regression(n_iterations=100, cross_val=5, num_jobs=-1)
        elif train_method == 'BayesHypOPt_Elanet_regression':
            result = self.BayesHypOPt_Elanet_regression(n_iterations=100, cross_val=5, num_jobs=-1)
        elif train_method == 'BayesHypOPt_GBM_regression':
            result = self.BayesHypOPt_GBM_regression(n_iterations=100, cross_val=5, num_jobs=-1)
        elif train_method == 'BayesHypOPt_HistGBM_regression':
            result = self.BayesHypOPt_HistGBM_regression(n_iterations=100, cross_val=5, num_jobs=-1)
        elif train_method == 'BayesHypOPt_SVR_regression':
            result = self.BayesHypOPt_SVR_regression(n_iterations=100, cross_val=5, num_jobs=-1)
        elif train_method == 'BayesHypOPt_NN_regression':
            result = self.BayesHypOPt_NN_regression(n_iterations=100, cross_val=5, num_jobs=-1)
        elif train_method == 'RandHypOPt_Ridge_regression':
            result = self.RandHypOPt_Ridge_regression(n_iterations=1000, cross_val=5, num_jobs=-1)
        elif train_method == 'RandHypOPt_Elanet_regression':
            result = self.RandHypOPt_Elanet_regression(n_iterations=1000, cross_val=5, num_jobs=-1)
        elif train_method == 'RandHypOPt_GBM_regression':
            result = self.RandHypOPt_GBM_regression(n_iterations=1000, cross_val=5, num_jobs=-1)
        elif train_method == 'RandHypOPt_SVR_regression':
            result = self.RandHypOPt_SVR_regression(n_iterations=1000, cross_val=5, num_jobs=-1)
        elif train_method == 'RandHypOPt_NN_regression':
            result = self.RandHypOPt_NN_regression(n_iterations=100, cross_val=5, num_jobs=-1)
        else:
            print('Undefined train method')
            return

        return result

    def BayesHypOPt_Ridge_regression(self, n_iterations=100, cross_val=5, num_jobs=-1):

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Setting the hyperparameters for Elastic Net regression
        Ridge_distributions = dict(alpha=Real(1, 1e+4, prior='uniform'))
        Ridge_training = BayesSearchCV(linear_model.Ridge(max_iter=1000),
                                       Ridge_distributions, n_iter=n_iterations,
                                       verbose=10, cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train)

        Ridge_model = Ridge_training.best_estimator_
        Ridge_model_cv = Ridge_training.cv_results_
        feature_importance_scores_ridge = pd.Series(list(Ridge_model.coef_), index=X_train.columns)
        y_test_predicted = Ridge_model.predict(X_test)
        y_train_predicted = Ridge_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (Ridge_model_cv['mean_test_score'][np.nanargmax(Ridge_model_cv['mean_test_score'])])
        std_cv_score = (Ridge_model_cv['std_test_score'][np.nanargmax(Ridge_model_cv['mean_test_score'])])
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_ridge, Ridge_model)
        return results

    def BayesHypOPt_Elanet_regression(self, n_iterations=100, cross_val=5, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        # Setting the hyperparameters for Elastic Net regression
        elanet_distributions = dict(alpha=Real(1e-3, 1, prior='uniform'),
                                    l1_ratio=Real(0, 1, prior='uniform'))

        elanet_training = BayesSearchCV(linear_model.ElasticNet(max_iter=1000), elanet_distributions,
                                        n_iter=n_iterations, verbose=10,
                                        cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train)

        Elanet_model = elanet_training.best_estimator_
        Elanet_model_cv = elanet_training.cv_results_
        feature_importance_scores_Elanet = pd.Series(list(Elanet_model.coef_), index=X_train.columns)
        y_test_predicted = Elanet_model.predict(X_test)
        y_train_predicted = Elanet_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (Elanet_model_cv['mean_test_score'][np.nanargmax(Elanet_model_cv['mean_test_score'])])
        std_cv_score = (Elanet_model_cv['std_test_score'][np.nanargmax(Elanet_model_cv['mean_test_score'])])
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_Elanet, Elanet_model)
        return results

    def BayesHypOPt_GBM_regression(self, n_iterations=100, cross_val=5, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Setting the hyperparameters for Gradient boosted decision trees
        GBM_distributions = dict(max_features=Categorical(["log2", "sqrt"]),
                                 # The number of features to consider when looking for the best split.
                                 learning_rate=Real(1e-4, 1, prior='uniform'),
                                 # Learning rate shrinks the contribution of each tree by learning_rate.
                                 subsample=Real(0.1, 1, prior='uniform'),
                                 # The fraction of samples to be used for fitting the individual base learners.
                                 min_samples_split=Integer(2, 100, prior='uniform'),
                                 # The minimum number of samples required to split an internal node.
                                 min_samples_leaf=Integer(2, 100, prior='uniform'),
                                 # The minimum number of samples required to be at a leaf node.
                                 n_estimators=Integer(100, 1000, prior='uniform'),
                                 # The number of boosting stages to perform.
                                 criterion=Categorical(['friedman_mse', 'squared_error']),
                                 # The function to measure the quality of a split.
                                 max_depth=Integer(2, 10, prior='uniform')
                                 # Maximum depth of the individual regression estimators.
                                 )

        GBM_training = BayesSearchCV(GradientBoostingRegressor(loss="squared_error"),
                                     GBM_distributions, n_iter=n_iterations, verbose=10,
                                     cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train)

        GBM_model = GBM_training.best_estimator_
        GBM_model_cv = GBM_training.cv_results_
        feature_importance_scores_GBM = pd.Series(list(GBM_model.feature_importances_), index=X_train.columns)
        y_test_predicted = GBM_model.predict(X_test)
        y_train_predicted = GBM_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
       # rmse_values = metrics.root_mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (GBM_model_cv['mean_test_score'][np.nanargmax(GBM_model_cv['mean_test_score'])])
        std_cv_score = (GBM_model_cv['std_test_score'][np.nanargmax(GBM_model_cv['mean_test_score'])])
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'cv_mean': mean_cv_score,
                  'cv_std': std_cv_score,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_GBM, GBM_model)
        return results



    def BayesHypOPt_HistGBM_regression(self, n_iterations=100, cross_val=5, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Setting the hyperparameters for Gradient boosted decision trees
        HistGBM_distributions = dict(

                                 learning_rate=Real(1e-4, 1, prior='uniform'),
                                 # Learning rate shrinks the contribution of each tree by learning_rate.
                                 max_leaf_nodes=Integer(2, 100, prior='uniform'),
                                 # The maximum number of leaves for each tree.
                                 max_depth=Integer(2, 10, prior='uniform'),
                                 # Maximum depth of the individual regression estimators.
                                 min_samples_leaf=Integer(2, 100, prior='uniform'),
                                 # The minimum number of samples required to be at a leaf node.
                                 max_bins=Integer(2, 255, prior='uniform')
                                # The maximum number of bins to use for non-missing values.
                                 )

        HistGBM_training = BayesSearchCV(HistGradientBoostingRegressor(loss="squared_error"),
                                     HistGBM_distributions, n_iter=n_iterations, verbose=10,
                                     cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train)

        HistGBM_model = HistGBM_training.best_estimator_
        HistGBM_model_cv = HistGBM_training.cv_results_
        #feature_importance_scores_GBM = pd.Series(list(HistGBM_model.feature_importances_), index=X_train.columns)
        y_test_predicted = HistGBM_model.predict(X_test)
        y_train_predicted = HistGBM_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        #rmse_values = metrics.root_mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (HistGBM_model_cv['mean_test_score'][np.nanargmax(HistGBM_model_cv['mean_test_score'])])
        std_cv_score = (HistGBM_model_cv['std_test_score'][np.nanargmax(HistGBM_model_cv['mean_test_score'])])
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'cv_mean': mean_cv_score,
                  'cv_std': std_cv_score,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, HistGBM_model)
        return results
    def BayesHypOPt_SVR_regression(self, n_iterations=100, cross_val=5, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Setting the hyperparameters for Support Vector regression.
        SVR_distributions = dict(C=Real(1e-5, 1.0, prior='log-uniform'),
                                 # C is the regularization parameter inversely proportional to the regularization strength.
                                 epsilon=Real(1e-2, 1.0, prior='log-uniform'),
                                 # Epsilon is the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value
                                 gamma=Categorical(["auto", "scale"]),
                                 # Gamma defines the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,and if ‘auto’, uses 1 / n_features.
                                 kernel=Categorical(["linear", "rbf"]))
                                 # Specifies the kernel type to be used in an algorithm.

        # Setting the hyperparameters for Gradient boosted decision trees
        SVR_training = BayesSearchCV(SVR(max_iter=1000,tol=1e-3), SVR_distributions, n_iter=n_iterations,
                                     cv=cross_val, n_jobs=num_jobs, verbose=10).fit(X_train, y_train)

        SVR_model = SVR_training.best_estimator_
        SVR_model_cv = SVR_training.cv_results_

        if SVR_model.kernel == 'linear':
            feature_importance_scores_SVR = pd.Series(list(SVR_model.coef_[0]), index=X_train.columns)
        else:
            feature_importance_scores_SVR = pd.Series('Nan')

        y_test_predicted = SVR_model.predict(X_test)
        y_train_predicted = SVR_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (SVR_model_cv['mean_test_score'][np.nanargmax(SVR_model_cv['mean_test_score'])])
        std_cv_score = (SVR_model_cv['std_test_score'][np.nanargmax(SVR_model_cv['mean_test_score'])])
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_SVR, SVR_model)
        return results

    def BayesHypOPt_NN_regression(self, n_iterations=100, cross_val=5, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Define a custom function to generate NN architectures with constraints
        def arch_hidden_layer_sizes(n_features, decrease_rate):

            architecture = []
            current_neurons = int(n_features)
            while current_neurons > 1:
                architecture.append(current_neurons)
                current_neurons = max(math.floor(current_neurons * decrease_rate), 1)
                print(current_neurons)
            return tuple(architecture)

        fixed_architecture = arch_hidden_layer_sizes(int(X_train.shape[1]), 0.75)
        NN_parameters = dict(hidden_layer_sizes=fixed_architecture,
                             activation=Categorical(["logistic", "relu"]),
                             solver=Categorical(["lbfgs", "adam"]),
                             alpha=Real(1e-5, 1, prior='log-uniform'),
                             batch_size=Integer(10, 50, prior='uniform'),
                             learning_rate_init=Real(1e-4, 1e-2, prior='log-uniform'))

        NN_training = BayesSearchCV(MLPRegressor(max_iter=100, verbose=True), NN_parameters,
                                    n_iter=n_iterations, cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train)

        NN_model = NN_training.best_estimator_
        NN_model_cv = NN_training.cv_results_
        y_test_predicted = NN_model.predict(X_test)
        y_train_predicted = NN_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (NN_model_cv['mean_test_score'][np.nanargmax(NN_model_cv['mean_test_score'])])
        std_cv_score = (NN_model_cv['std_test_score'][np.nanargmax(NN_model_cv['mean_test_score'])])
        feature_importance_scores_NN = 'Nan'
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_NN, NN_model)
        return results


    def RandHypOPt_Ridge_regression(self, n_iterations=1000, cross_val=5, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        Ridge_distributions = dict(alpha=loguniform(1, 1e3))
                                   # Regularization parameter.
        Ridge_training = RandomizedSearchCV(linear_model.Ridge(max_iter=10000),
                                            # Maximum iterations in case the model doesn't converge before.
                                            Ridge_distributions, n_iter=n_iterations, verbose=10,
                                            cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train)

        Ridge_model = Ridge_training.best_estimator_
        Ridge_model_cv = Ridge_training.cv_results_
        feature_importance_scores_ridge = pd.Series(list(Ridge_model.coef_), index=X_train.columns)
        y_test_predicted = (Ridge_model.predict(X_test))
        y_train_predicted = (Ridge_model.predict(X_train))
        test_r2score = (metrics.r2_score(y_test, y_test_predicted))
        train_r2score = (metrics.r2_score(y_train, y_train_predicted))
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = (stats.pearsonr(y_test, y_test_predicted)[0])
        test_pears_pval = (stats.pearsonr(y_test, y_test_predicted)[1])
        train_pears_val = (stats.pearsonr(y_train, y_train_predicted)[0])
        train_pears_pval = (stats.pearsonr(y_train, y_train_predicted)[1])
        mean_cv_score = (Ridge_model_cv['mean_test_score'][np.nanargmax(Ridge_model_cv['mean_test_score'])])
        std_cv_score = (Ridge_model_cv['std_test_score'][np.nanargmax(Ridge_model_cv['mean_test_score'])])

        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_ridge, Ridge_model)
        return results

    def RandHypOPt_Elanet_regression(self, n_iterations=1000, cross_val=3, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Setting the hyperparameters for Elastic Net regression
        elanet_distributions = dict(alpha=uniform(1e-3, 1),
                                    # Regularization parameter
                                    l1_ratio=uniform(0.0, 1.0)
                                    # Parameter controlling the weights of the contribution of the L1 and L2 regularization.
                                    )

        Elanet_training = RandomizedSearchCV(linear_model.ElasticNet(max_iter=10000), elanet_distributions,
                                             n_iter=n_iterations, verbose=10, cv=cross_val,
                                             n_jobs=num_jobs).fit(X_train, y_train)

        Elanet_model = Elanet_training.best_estimator_
        Elanet_model_cv = Elanet_training.cv_results_
        feature_importance_scores_Elanet = pd.Series(list(Elanet_model.coef_), index=X_train.columns)
        y_test_predicted = np.array(Elanet_model.predict(X_test))
        y_train_predicted = np.array(Elanet_model.predict(X_train))
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (Elanet_model_cv['mean_test_score'][np.nanargmax(Elanet_model_cv['mean_test_score'])])
        std_cv_score = (Elanet_model_cv['std_test_score'][np.nanargmax(Elanet_model_cv['mean_test_score'])])
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_Elanet, Elanet_model)
        return results

    def RandHypOPt_GBM_regression(self, n_iterations=1000, cross_val=5, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Setting the hyperparameters for Gradient boosted decision trees
        GBM_distributions = dict(max_features=["auto", "log2", "sqrt"], learning_rate=uniform(1e-3, 1),
                                 max_depth=randint(2, 10), subsample=uniform(0, 1),
                                 min_samples_split=randint(2, 100), min_samples_leaf=randint(2, 100),
                                 n_estimators=randint(4, 100), criterion=['friedman_mse', 'squared_error'])

        GBM_training = RandomizedSearchCV(GradientBoostingRegressor(loss="squared_error", n_iter_no_change=5),
                                          GBM_distributions, n_iter=n_iterations, verbose=10,
                                          cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train)

        GBM_model = GBM_training.best_estimator_
        GBM_model_cv = GBM_training.cv_results_
        feature_importance_scores_GBM = pd.Series(list(GBM_model.feature_importances_), index=X_train.columns)
        y_test_predicted = GBM_model.predict(X_test)
        y_train_predicted = GBM_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (GBM_model_cv['mean_test_score'][np.nanargmax(GBM_model_cv['mean_test_score'])])
        std_cv_score = (GBM_model_cv['std_test_score'][np.nanargmax(GBM_model_cv['mean_test_score'])])
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_GBM, GBM_model)
        return results

    def RandHypOPt_SVR_regression(self, n_iterations=1000, cross_val=5, num_jobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Setting the hyperparameters for Support Vector regression
        SVR_distributions = dict(C=loguniform(1e-3, 1e-2),
                                 # C is the regularization parameter inversely proportional to the regularization strength.
                                 epsilon=loguniform(1e-1, 1),
                                 # Epsilon is the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value
                                 gamma=["auto", "scale"],
                                 # Gamma defines the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,and if ‘auto’, uses 1 / n_features.
                                 kernel=["linear", "rbf"]
                                 # Specifies the kernel type to be used in an algorithm.
                                 )

        SVR_training = RandomizedSearchCV(SVR(max_iter=1000,tol=1e-3), SVR_distributions, n_iter=n_iterations,
                                          cv=cross_val, n_jobs=num_jobs, verbose=10).fit(X_train,y_train)

        SVR_model = SVR_training.best_estimator_
        SVR_model_cv = SVR_training.cv_results_
        if SVR_model.kernel=='linear':
            feature_importance_scores_SVR = pd.Series(list(SVR_model.coef_[0]), index=X_train.columns)
        else:
            feature_importance_scores_SVR = pd.Series('Nan')
        y_test_predicted = SVR_model.predict(X_test)
        y_train_predicted = SVR_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        mean_cv_score = (SVR_model_cv['mean_test_score'][np.nanargmax(SVR_model_cv['mean_test_score'])])
        std_cv_score = (SVR_model_cv['std_test_score'][np.nanargmax(SVR_model_cv['mean_test_score'])])
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval,
                  'CV_best_mean_score': mean_cv_score,
                  'CV_mean_std_score': std_cv_score}
        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_SVR, SVR_model)
        return results

    def RandHypOPt_NN_regression(self, n_iterations=1000, cross_val=3, num_jobs=-1):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint, loguniform
        from sklearn.neural_network import MLPRegressor
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Define a custom function to generate NN architectures with constraints
        def arch_hidden_layer_sizes(n_features, decrease_rate):

            architecture = []
            current_neurons = int(n_features)
            while current_neurons > 1:
                architecture.append(current_neurons)
                current_neurons = max(math.floor(current_neurons * decrease_rate), 1)
                print(current_neurons)
            return tuple(architecture)

        fixed_architecture = arch_hidden_layer_sizes(int(X_train.shape[1]), 0.75)

        NN_parameters = dict(hidden_layer_sizes=fixed_architecture,
                             activation=Categorical(["logistic", "relu"]),
                             solver=Categorical(["lbfgs", "adam"]),
                             alpha=loguniform(0.1, 0.9),
                             batch_size=randint(10, 50),
                             learning_rate_init=loguniform(1e-4, 1e-2))

        NN_training = RandomizedSearchCV(MLPRegressor(max_iter=100, verbose=True),
                                                   NN_parameters, n_iter=n_iterations, cv=cross_val,
                                                   n_jobs=num_jobs).fit(X_train, y_train)
        NN_model = NN_training.best_estimator_
        y_test_predicted = NN_model.predict(X_test)
        y_train_predicted = NN_model.predict(X_train)
        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        mse_values = metrics.mean_squared_error(y_test, y_test_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        feature_importance_scores_NN = 'Nan'
        scores = {'Test r2score': test_r2score,
                  'Train r2 score': train_r2score,
                  'mse': mse_values,
                  'Test pearson value': test_pears_val,
                  'Test pearson p-value': test_pears_pval,
                  'Train pearson value': train_pears_val,
                  'Train pearson p-value': train_pears_pval}

        scores = pd.Series(list(scores.values()), index=(scores.keys()))
        results = PredictionResults(scores, y_train_predicted, y_test_predicted, feature_importance_scores_NN, NN_model)

        return results
