import pandas as pd
import os
import json
import numpy as np
import time
from model import ModelBuilder
from parameters import get_parameters
from preprocess import DataPreprocessing
from feature_selection import FeatureSelection

start_time = time.time()
if __name__ == '__main__':
    params = get_parameters()
    data = pd.read_csv(params.data_path, index_col=0)
    file_name = os.path.splitext(os.path.basename(params.data_path))[0]

    data_preprocessor = DataPreprocessing(data)
    if params.data_splitting_criteria == 'preprocess_data_HOAR':
        preprocessed_data = data_preprocessor.preprocess_data_HOAR(test_split_size=0.25)
    elif params.data_splitting_criteria == 'preprocess_data_INHO':
        clades = pd.read_csv(params.clades_data_path, index_col=0)
        preprocessed_data = data_preprocessor.preprocess_data_INHO(clades)
    elif params.data_splitting_criteria == 'preprocess_data_LOCO':
        clades = pd.read_csv(params.clades_data_path, index_col=0)
        preprocessed_data = data_preprocessor.preprocess_data_LOCO(clades)

    if params.do_feature_selection:
        features_selector = FeatureSelection(preprocessed_data['X_train'],
                                             preprocessed_data['X_test'],
                                             preprocessed_data['y_train'])
        features = features_selector.select_features(method=params.feature_selection_strategy)
    else:
        features = preprocessed_data

    model = ModelBuilder(X_train=features['X_train'], X_test=features['X_test'],
                         y_train=preprocessed_data['y_train'], y_test=preprocessed_data['y_test'])

    def stable_feature_importance(model, model_type, n_runs=10):
        importances = []
        Test_r2score = []
        Train_r2score = []
        MSE = []
        CV_mean_score = []
        CV_score_std = []
        Test_pears_val = []
        Train_pears_val = []
        Test_pears_pval = []
        Train_pears_pval = []

        for i in range(n_runs):
            try:
                r = model.train_model(train_method=model_type)
                Test_r2score.append(r.results['Test r2score'])
                Train_r2score.append(r.results['Train r2 score'])
                MSE.append(r.results['mse'])
                CV_mean_score.append(r.results['cv_mean'])
                CV_score_std.append(r.results['cv_std'])
                Test_pears_val.append(r.results['Test pearson value'])
                Train_pears_val.append(r.results['Train pearson value'])
                Test_pears_pval.append(r.results['Test pearson p-value'])
                Train_pears_pval.append(r.results['Train pearson p-value'])
                importances.append(r.feature_importance_scores)
                Results = {'Mean Test r2score': np.mean(Test_r2score,axis=0),
                           'Mean Train r2score': np.mean(Train_r2score, axis=0),
                           'Mean MSE': np.mean(MSE, axis=0),
                           'Mean cv score': np.mean(CV_mean_score, axis=0),
                           'Mean Test Pearson': np.mean(Test_pears_val,axis=0),
                           'Mean Train Pearson': np.mean(Train_pears_val,axis=0),
                           'Mean FI scores': np.mean(importances,axis=0)
                           }
            except Exception as e:
                print(e)
        return Results


    FIscores = stable_feature_importance(model,params.model_type,10)
    with open(f'{params.data_path_out}/{file_name}_{params.model_type}_10iterartions_results.json', 'w+') as f:
        d={'Test_r2score': FIscores['Mean Test r2score'],
             'Train_r2score': FIscores['Mean Train r2score'],
             'MSE': FIscores['Mean MSE'],
             'CV mean score': FIscores['Mean cv score'],
             'Test_pearson_value': FIscores['Mean Test Pearson'],
             'Train_pearson_value': FIscores['Mean Train Pearson']

        }
        json.dump(d, f)
    df=  pd.DataFrame(FIscores['Mean FI scores'],features['X_train'].columns)
    df.to_csv(f'{params.data_path_out}/{file_name}_{params.model_type}_10iterartions_FIscores.csv')