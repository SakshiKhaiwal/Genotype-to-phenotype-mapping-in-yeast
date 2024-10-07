import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


# Loading data

Predictor_matrix = pd.read_csv('path_to_predictor_matrix.csv')
Target_matrix = pd.read_csv('path_to_target_matrix.csv')

# Preprocessing the data


Target_matrix_mean = Target_matrix.mean()
Target_matrix_processed = Target_matrix.fillna(Target_matrix_mean)
sss = ShuffleSplit(n_splits=1, test_size=0.25)
sss.get_n_splits(Predictor_matrix, Target_matrix_processed)
train_index, test_index = next(sss.split(Predictor_matrix, Target_matrix_processed))
X_train, X_test = Predictor_matrix.iloc[train_index, :], Predictor_matrix.iloc[test_index, :]
y_train, y_test = Target_matrix_processed.iloc[train_index, :], Target_matrix_processed.iloc[test_index, :]
training_strains, testing_strains = Predictor_matrix.index[train_index], Predictor_matrix.index[test_index]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
X_test = scaler.fit_transform(X_test.values)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)


# Multitarget GBM regression

n_iterations = 100
cross_val = 3
num_jobs = -1

GBM_distributions = dict(max_features=["auto", "log2", "sqrt"], learning_rate=uniform(1e-3, 1),
                         subsample=uniform(0, 1),
                         min_samples_split=randint(2, 100), min_samples_leaf=randint(2, 100),
                         n_estimators=randint(4, 100), criterion=['friedman_mse', 'squared_error'],
                         max_depth=randint(2, 10))
multireg = MultiOutputRegressor(RandomizedSearchCV(GradientBoostingRegressor(loss="squared_error", n_iter_no_change=5), # To decide if early stopping will be used to terminate training when validation score is not improving.
                                                   GBM_distributions, n_iter=n_iterations, verbose=10,
                                                   cv=cross_val, n_jobs=num_jobs)).fit(X_train, y_train)
multi_reg_pred = multireg.predict(X_test)
y_test_vec = y_test.flatten()
multi_reg_pred_vec = multi_reg_pred.flatten()

pd.DataFrame(y_test,index=testing_strains,columns=Target_matrix.columns).to_csv('y_test_multitarget_RandGBM_LOF.csv')
pd.DataFrame(multi_reg_pred,index=testing_strains,columns=Target_matrix.columns).to_csv('y_test_predicted_multitarget_RandGBM_LOF.csv')


feature_importances = []

# Loop through each estimator in the MultiOutputRegressor
for estimator in multireg.estimators_:
    # The estimator here is the RandomizedSearchCV object
    # Access the best estimator from RandomizedSearchCV
    best_estimator = estimator.best_estimator_

    # Now, extract the feature importances from the best_estimator,
    # which is an instance of GradientBoostingRegressor
    importances = best_estimator.feature_importances_
    feature_importances.append(importances)



feature_names = Predictor_matrix.columns
importances_df = pd.DataFrame(feature_importances).T
importances_df.columns = Target_matrix.columns
importances_df.index = feature_names
importances_df.to_csv('feature_importances_LOF_multitarget_RandGBM.csv')

