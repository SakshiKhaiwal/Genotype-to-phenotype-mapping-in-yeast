import numpy as np
import functools
import operator
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

#  This is the section on data preprocessing which includes, splitting the test and training set,
#  removing NA and scaling the data on a normal scale.
#  We have implemented four strategies used for our purpose in the paper:
#  1. HOAR: Hold-out at random, randomly splitting the entire population into 75% training and 25% testing set.
#  2. INHO: Intra-clade hold out, considering only one clade (Wine European) and keeping 50 strains in testing and rest for training.
#  3. LOCO: Leave-one clade out, use all clades for training except one that is used as a testing set.


class DataPreprocessing:

    def __init__(self, data):
        #  This is the data preprocessing section.
        #  The data is input as csv file with first column as phenotype (target) and the rest with features.
        #  The row names should be the name of the strains to return strains in test and training set.

        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        #  Removing data with missing labels
        self.X = pd.DataFrame(self.data.iloc[:, 1:])
        self.y = pd.DataFrame(self.data.iloc[:, 0])
        self.strains = pd.DataFrame(data.index, index=data.index)
        self.missing_values_strains = [index for index, row in self.y.iterrows() if row.isnull().any()]
        self.X = self.X.drop(self.missing_values_strains, axis=0, inplace=False)
        self.y = self.y.drop(self.missing_values_strains, axis=0, inplace=False)
        self.strains = self.strains.drop(self.missing_values_strains, axis=0, inplace=False)
        # Filter out the features with more than 25% of NA values.
        threshold_for_nan = 0.25
        self.X = self.X.loc[:, self.X.isna().mean() <= threshold_for_nan]


    def preprocess_data_HOAR(self, test_split_size=0.25,impute_method='MI'):

        """
        This function splits the data into training and testing and performs the preprocessing on them individually.
        :param test_split_size: The fraction of data to be kept for the testing set.
        :return: The features and target data for training and testing set and the corresponding strains in both.
        """
        #  Shuffle and split the data randomly into a training and testing set.
        sss = ShuffleSplit(n_splits=1, test_size=test_split_size)
        sss.get_n_splits(self.X, self.y,)
        train_index, test_index = next(sss.split(self.X, self.y))
        self.X_train, self.X_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
        self.y_train, self.y_test = self.y.iloc[train_index, :], self.y.iloc[test_index, :]
        strains_training, strains_testing = self.strains.iloc[train_index, :], self.strains.iloc[test_index, :]


        if (impute_method=='MI'):

            X_train_means = self.X_train.mean()
            X_test_means = self.X_test.mean()

            X_train = self.X_train.fillna(X_train_means)
            X_test = self.X_test.fillna(X_test_means)
        elif (impute_method=='KNN'):
            knn_imputer = KNNImputer(n_neighbors=6)

            # Fit and transform the training data
            X_train = pd.DataFrame(knn_imputer.fit_transform(self.X_train.values), columns=self.X_train.columns,
                                   index=self.X_train.index)
            X_test = pd.DataFrame(knn_imputer.fit_transform(self.X_test), columns=self.X_test.columns,
                                  index=self.X_test.index)


        scaler = StandardScaler().fit(self.X_train.values)
        self.X_train = pd.DataFrame(scaler.transform(self.X_train.values), columns=self.X_train.columns, index=self.X_train.index)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test.values), columns=self.X_test.columns, index=self.X_test.index)

        y_test = np.asarray(functools.reduce(operator.iconcat, np.asarray(self.y_test), []))
        y_train = np.asarray(functools.reduce(operator.iconcat, np.asarray(self.y_train), []))
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
                'strains_training': strains_training, 'strains_testing': strains_testing}


    def preprocess_data_INHO(self, clades):

        """
        This function allows you to choose only one clade for performing the predictions.
        In our case, we used Wine European as it is the clade with the largest number of strains.
        The training and testing set are then preprocessed individually.

        :param clades:  clades classification for the strains.
        :return: The features and target data for training and testing set and the corresponding strains in both.
        """
        self.clades = clades
        clades_filter = self.clades.drop(self.missing_values_strains, axis=0, inplace=False)
        WE_index = np.where(clades_filter == '01.Wine_European')[0]     ### Choose the clade to keep. In this case, we are keeping only the Wine European clade.
        test_index = np.random.choice(WE_index, size=50, replace=False) ### Randomly choosing 50 strains for testing.
        train_index = WE_index[~np.in1d(WE_index, test_index)]          ###Choosing rest of the strains for training.
        X_train, X_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
        y_train, y_test = self.y.iloc[train_index, :], self.y.iloc[test_index, :]
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.fit_transform(X_test.values), columns=X_test.columns, index=X_test.index)
        X_train_means = X_train.mean()
        X_test_means = X_test.mean()
        X_train = X_train.fillna(X_train_means)
        X_test = X_test.fillna(X_test_means)
        y_test = np.asarray(functools.reduce(operator.iconcat, np.asarray(y_test), []))
        y_train = np.asarray(functools.reduce(operator.iconcat, np.asarray(y_train), []))
        test_clades = self.clades.values[test_index]
        train_clades = self.clades.values[train_index]
        strains_training, strains_testing = self.strains.iloc[train_index, :], self.strains.iloc[test_index, :]
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
                'test_clades': test_clades, 'train_clades': train_clades,
                'strains_training': strains_training, 'strains_testing': strains_testing}


    def preprocess_data_LOCO(self, clades):

        """
        This function allows you to perform leave-one-clade out validation.
        The training and testing set are then preprocessed individually.

        :param clades:  clades classification for the strains.
        :return: The features and target data for training and testing set and the corresponding strains in both.
        """
        self.clades = clades
        clades_filter = self.clades.drop(self.missing_values_strains, axis=0, inplace=False)
        test_index = np.where(clades_filter == 'M3.Mosaic_Region_3')[0]        ### Randomly choosing 50 strains for testing.
        train_index = np.where(clades_filter != 'M3.Mosaic_Region_3')[0]       ###Choosing rest of the strains for training.
        X_train, X_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
        y_train, y_test = self.y.iloc[train_index, :], self.y.iloc[test_index, :]
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.fit_transform(X_test.values), columns=X_test.columns, index=X_test.index)
        X_train_means = X_train.mean()
        X_test_means = X_test.mean()
        X_train = X_train.fillna(X_train_means)
        X_test = X_test.fillna(X_test_means)
        y_test = np.asarray(functools.reduce(operator.iconcat, np.asarray(y_test), []))
        y_train = np.asarray(functools.reduce(operator.iconcat, np.asarray(y_train), []))
        test_clades = self.clades.values[test_index]
        train_clades = self.clades.values[train_index]
        strains_training, strains_testing = self.strains.iloc[train_index, :], self.strains.iloc[test_index, :]
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
                'test_clades': test_clades, 'train_clades': train_clades,
                'strains_training': strains_training, 'strains_testing': strains_testing}



