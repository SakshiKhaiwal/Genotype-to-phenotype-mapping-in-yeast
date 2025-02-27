import pandas as pd
import os
import json
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
    start_time = time.time()
    r = model.train_model(train_method=params.model_type)
    end_time = time.time()
    training_time = end_time - start_time

    with open(f'{params.data_path_out}/{file_name}_{params.model_type}.json', 'w+') as f:
        d = {'Test_r2score': r.results['Test r2score'],
             'Train_r2score': r.results['Train r2 score'],
             'MSE': r.results['mse'],
             'CV mean score': r.results['cv_mean'],
             'CV std': r.results['cv_std'],
             'Test_pearson_value': r.results['Test pearson value'],
             'Train_pearson_value': r.results['Train pearson value'],
             'Test_pearson_pvalue': r.results['Test pearson p-value'],
             'Train_pearson_pvalue': r.results['Train pearson p-value'],
             'Training_time': training_time}
        json.dump(d, f)

    with open(f'{params.data_path_out}/{file_name}_{params.model_type}_additional_information.json', 'w+') as f:
        d = {'y_train_predicted': r.y_train_predicted,
             'y_test_predicted': r.y_test_predicted,
             'Features importance scores': r.feature_importance_scores.to_dict()}
        json.dump(d, f)

