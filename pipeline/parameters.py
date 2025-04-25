import argparse


def get_parameters():

    """
    Define parameters to run the prediction, including the input and output data path, splitting criteria,
    feature selection strategy and model type.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the input data')
    parser.add_argument('--data_path_out', type=str, required=True,
                        help='path to the output data')
    parser.add_argument('--clades_data_path', type=str, required=False, help='path to clades data')
    parser.add_argument('--data_splitting_criteria', type=str, default='preprocess_data_HOAR_MI',
                        choices=['preprocess_data_HOAR_MI','preprocess_data_HOAR_KNN','preprocess_data_INHO', 'preprocess_data_LOCO'],
                        help='how to split the data into training and testing')
    parser.add_argument('--do_feature_selection', type=bool, required=False, default=False,
                        help='apply feature selection')
    parser.add_argument('--feature_selection_strategy', type=str, required=False, default='lasso_selection_grid',
                        choices=['lasso_selection_grid', 'lasso_selection_random',
                                 'lasso_selection_bayes', 'Boruta_selection','high_lasso'],
                        help='choice of feature selection strategy')
    parser.add_argument('--model_type', type=str, default='BayesHypOPt_GBM_regression',
                        choices=['BayesHypOPt_Ridge_regression', 'BayesHypOPt_Elanet_regression',
                                 'BayesHypOPt_GBM_regression','BayesHypOPt_HistGBM_regression',
                                 'BayesHypOPt_SVR_regression','BayesHypOPt_NN_regression',
                                 'RandHypOPt_Ridge_regression','RandHypOPt_Elanet_regression',
                                 'RandHypOPt_GBM_regression','RandHypOPt_SVR_regression',
                                 'RandHypOPt_NN_regression'], help='used model')
    parser.add_argument('--n_iterations', type=int, default=100,
                        help='number of iterations for the model')
    parser.add_argument('--cross_val', type=int, default=5,
                        help='value of the cross validation')
    parser.add_argument('--num_jobs', type=int, default=-1,
                        help='number of jobs to use')

    return parser.parse_args()
