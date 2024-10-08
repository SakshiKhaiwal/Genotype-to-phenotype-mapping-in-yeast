import json
import os
import time
import pandas as pd
from model import ModelBuilder
from parameters import get_parameters
from preprocess import DataPreprocessing
from feature_selection import FeatureSelection

if __name__ == '__main__':
    params = get_parameters()


    GenMatrices = [f for f in os.listdir(params.data_path) if f.endswith('gen.csv')] # List of all genetic matrices
    PhenMatrix = [f for f in os.listdir(params.data_path) if f.endswith('phen.csv')] # Phenotypic matrix

    y_data = pd.read_csv(os.path.join(params.data_path, PhenMatrix), index_col=0)
    num_targets = y_data.shape[1]

    models_to_train = ['RandHypOPt_Ridge_regression', 'RandHypOPt_GBM_regression',
                       'RandHypOPt_SVR_regression', 'RandHypOPt_NN_regression',
                       'BayesHypOPt_Ridge_regression', 'BayesHypOPt_GBM_regression',
                       'BayesHypOPt_SVR_regression', 'BayesHypOPt_NN_regression']
    exception_list = []

    for x_name in GenMatrices:
        Phenotypes_with_error = []
        Phenotypes_predicted = []
        features = pd.read_csv(os.path.join(params.data_path, x_name), index_col=0)

        for i in range(0, num_targets):
            Test_r2score = []
            Train_r2score = []
            Test_pears_val = []
            Train_pears_val = []
            Test_pears_pval = []
            Train_pears_pval = []
            y_train_predicted = {}
            y_test_predicted = {}
            training_strains = {}
            testing_strains = {}
            feature_importance_scores = {}
            target = y_data.iloc[:, i]
            Phenotype_name = target.name
            data = pd.merge(target, features, left_index=True, right_index=True)

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
                selected_features = features_selector.select_features(method=params.feature_selection_strategy)
            else:
                selected_features = preprocessed_data

            models_trained = []
            models_training_times = []
            for model_name in models_to_train:
                try:
                    model = ModelBuilder(X_train=selected_features['X_train'], X_test=selected_features['X_test'],
                                         y_train=preprocessed_data['y_train'], y_test=preprocessed_data['y_test'])
                    start_time = time.time()
                    r = model.train_model(train_method=model_name)
                    end_train = time.time()
                    training_time = end_train-start_time
                    models_trained.append(model_name)
                    Test_r2score.append(r.results['Test r2score'])
                    Train_r2score.append(r.results['Train r2 score'])
                    Test_pears_val.append(r.results['Test pearson value'])
                    Train_pears_val.append(r.results['Train pearson value'])
                    Test_pears_pval.append(r.results['Test pearson p-value'])
                    Train_pears_pval.append(r.results['Train pearson p-value'])
                    models_training_times.append(training_time)
                    testing_strains[model_name] = preprocessed_data['strains_testing'].iloc[:, 0].tolist()
                    y_test_predicted[model_name] = r.y_test_predicted.tolist()
                    training_strains[model_name] = preprocessed_data['strains_testing'].iloc[:, 0].tolist()
                    y_train_predicted[model_name] = r.y_train_predicted.tolist()
                    try:
                        feature_importance_scores[model_name] = r.feature_importance_scores.to_dict()
                    except Exception as e:
                        print(e)
                        feature_importance_scores[model_name] = 'NA'
                except Exception as e:
                    print(e)
                    Phenotypes_with_error.append(Phenotype_name)

            with open(f'{params.data_path_out}/{x_name}_{Phenotype_name}.json', 'w+') as f:
                d = {
                    'models used': models_trained,
                    'Test r2 score': Test_r2score,
                    'Train r2 score': Train_r2score,
                    'Test pears value': Test_pears_val,
                    'Train pears value': Train_pears_val,
                    'Train p-value': Train_pears_pval,
                    'Test p-value': Test_pears_pval,
                    'Training time': training_time
                }

                json.dump(d, f)

            with open(f'{params.data_path_out}/{x_name}_{Phenotype_name}_with_additional_information.json',
                      'w+') as f:
                d = {'y_train_predicted': y_train_predicted,
                     'y_test_predicted': y_test_predicted,
                     'training_strains': training_strains,
                     'testing_strains': testing_strains,
                     'Features importance scores': feature_importance_scores
                     }
                json.dump(d, f)
        (pd.DataFrame(Phenotypes_with_error)).to_csv(f'{x_name}_Phenotypes_with_error.csv')