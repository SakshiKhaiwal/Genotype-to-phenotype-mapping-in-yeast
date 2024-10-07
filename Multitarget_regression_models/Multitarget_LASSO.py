
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
import time
from sklearn.model_selection import GridSearchCV


### Loading data

Predictor_matrix = pd.read_csv('path_to_predictor_matrix.csv')
Target_matrix = pd.read_csv('path_to_target_matrix.csv')


### Data preprocessing

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


##### Multitarget LASSO regression

lasso = linear_model.MultiTaskLasso(random_state=0, max_iter=10000)
alpha = (0.5, 0.75, 1)
tuned_parameters = [{'alpha': alpha}]
start_training = time.time()
model_fit= GridSearchCV(lasso, tuned_parameters, cv=3 , n_jobs=-1, refit=False).fit(X_train, y_train)
best_parameter= model_fit.best_params_
model = linear_model.MultiTaskLasso(best_parameter['alpha'], random_state=0, max_iter=10000).fit(X_train, y_train)
end_training = time.time()
training_time = end_training - start_training


predicted_ytest = model.predict(X_test)
pd.DataFrame(predicted_ytest, columns=Target_matrix_processed.columns, index=testing_strains).to_csv('y_test_predicted.csv')
pd.DataFrame(y_test, columns=Target_matrix_processed.columns, index=testing_strains).to_csv('y_test_true.csv')

with open(f'Modelparam_and_trainingtimes_LOF', 'w+') as f:
    d = {'best_param': best_parameter,
         'Training time': training_time
         }
    json.dump(d, f)

pd.DataFrame(model.coef_, index=Target_matrix_processed.columns, columns=Predictor_matrix.columns).to_csv('MultiTASKLASSO_feature_importance.csv')

