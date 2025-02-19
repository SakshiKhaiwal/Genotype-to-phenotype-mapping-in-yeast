import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.feature_selection import SelectFromModel
from skopt import BayesSearchCV
from skopt.space import Real
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from hi_lasso.hi_lasso import HiLasso



class FeatureSelection:

    def __init__(self, X_train, X_test, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.selector = None

    def select_features(self, method='lasso_selection_grid'):
        if method == 'lasso_selection_grid':
            features = self.lasso_selection_grid()
        elif method == 'high_lasso':
            features = self.high_lasso()
        elif method == 'lasso_selection_random':
            features = self.lasso_selection_random()
        elif method == 'lasso_selection_bayes':
            features = self.lasso_selection_bayes()
        elif method == 'Boruta_selection':
            features = self.Boruta_selection()

        else:
            print('unrecognized feature selection method')
            return

        return features

    def lasso_selection_grid(self, nfolds=5, njobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train

        lasso = Lasso(random_state=0, max_iter=10000)
        alphas = (0.001, 0.01, 0.1, 0.5)
        tuned_parameters = [{'alpha': alphas}]
        Tuned_lasso_alpha = GridSearchCV(lasso, tuned_parameters, cv=nfolds, n_jobs=njobs, refit=False).fit(X_train, y_train).best_params_
        selector = SelectFromModel(estimator=Lasso(random_state=1000, max_iter=50000,
                                                   alpha=Tuned_lasso_alpha['alpha'])).fit(X_train, y_train)
        LASSO_X_train = selector.transform(X_train)
        LASSO_X_test = selector.transform(X_test)
        LASSO_X_train = pd.DataFrame(data=LASSO_X_train,
                                     columns=X_train.columns[np.where(selector.get_support()==True)[0]])
        LASSO_X_test = pd.DataFrame(data=LASSO_X_test,
                                    columns=X_test.columns[np.where(selector.get_support()==True)[0]])

        return {'X_train': LASSO_X_train, 'X_test': LASSO_X_test}

    def lasso_selection_random(self, n_iterations=500, nfolds=5, njobs=-1):

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train

        lasso = Lasso(random_state=0, max_iter=10000)
        tuned_parameters = dict(alpha=loguniform(1e-4, 1))
        Tuned_lasso_alpha = RandomizedSearchCV(lasso,  tuned_parameters, n_iter=n_iterations,
                                               cv=nfolds, n_jobs=njobs, refit=False).fit(X_train, y_train).best_params_
        selector = SelectFromModel(estimator=Lasso(random_state=1000, max_iter=50000,
                                                   alpha=Tuned_lasso_alpha['alpha'])).fit(X_train, y_train)
        LASSO_X_train = selector.transform(X_train)
        LASSO_X_test = selector.transform(X_test)
        LASSO_X_train = pd.DataFrame(data=LASSO_X_train,
                                     columns=X_train.columns[np.where(selector.get_support()==True)[0]])
        LASSO_X_test = pd.DataFrame(data=LASSO_X_test,
                                    columns=X_test.columns[np.where(selector.get_support()==True)[0]])

        return {'X_train': LASSO_X_train, 'X_test': LASSO_X_test}

    def lasso_selection_bayes(self, n_iterations=200, nfolds=5, njobs=-1):

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train

        lasso = Lasso(random_state=0, max_iter=10000)
        tuned_parameters = dict(alpha=Real(1e-4, 1, prior='log-uniform'))
        Tuned_lasso_alpha = BayesSearchCV(lasso, tuned_parameters, n_iter=n_iterations, cv=nfolds,
                                          n_jobs=njobs, refit=False).fit(X_train, y_train).best_params_
        selector = SelectFromModel(estimator=Lasso(random_state=1000, max_iter=50000,
                                                   alpha=Tuned_lasso_alpha['alpha'])).fit(X_train, y_train)
        LASSO_X_train = selector.transform(X_train)
        LASSO_X_test = selector.transform(X_test)
        LASSO_X_train = pd.DataFrame(data=LASSO_X_train,
                                     columns=X_train.columns[np.where(selector.get_support()==True)[0]])
        LASSO_X_test = pd.DataFrame(data=LASSO_X_test,
                                    columns=X_test.columns[np.where(selector.get_support()==True)[0]])

        return {'X_train': LASSO_X_train, 'X_test': LASSO_X_test}



    def Boruta_selection(self, n_iterations=200, nfolds=5, njobs=-1):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train

        rf = RandomForestRegressor(random_state=0, n_estimators=500)
        boruta = BorutaPy(rf, n_estimators='auto', max_iter=1000,alpha=0.5, verbose=2, random_state=42)
        boruta.fit(X_train.values, y_train)
        #index_selected_features = np.where(boruta.support_==True)[0]
        Boruta_X_train = boruta.transform(X_train.values)
        Boruta_X_test =  boruta.transform(X_test.values)
        Boruta_X_train = pd.DataFrame(data=Boruta_X_train,
                                     columns=X_train.columns[np.where(boruta.support_==True)[0]])
        Boruta_X_test = pd.DataFrame(data=Boruta_X_test,
                                    columns=X_test.columns[np.where(boruta.support_==True)[0]])

        return {'X_train': Boruta_X_train, 'X_test': Boruta_X_test}
    def high_lasso(self, l=30, alpha=0.05, njobs=50):

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        hilasso = HiLasso(q1=175, q2=175, L=l, alpha=alpha, logistic=False, random_state=None, parallel=True, n_jobs=njobs)
        hilasso.fit(X_train, y_train, sample_weight=None)
        hilasso_coef = hilasso.coef_
        X_train_HiLasso = X_train.iloc[:, np.where(hilasso_coef)[0]]
        X_test_HiLasso = X_test.iloc[:, np.where(hilasso_coef)[0]]
        return {'X_train': X_train_HiLasso, 'X_test': X_test_HiLasso}


