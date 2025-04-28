# Genotype to phenotype prediction of natural variation in **S. cerevisiae**.

In this project, we investigate the associations between the phenotypes and genotypes in a natural yeast population. We have accumulated >200 phenotypes from various studies in addition to 33 unpublished phenotypes for 1011 **cerevisiae** strains. To construct meaningful models that can provide predictions of the phenotype in a novel population based on its genotypes, we explored several linear and nonlinear machine learning (ML) methods. Since we are interested in the prediction of quantitative phenotypes, we utilized regression-based methods, such as ridge regression, elastic net regression, support vector regression (SVR), gradient-boosted machines (GBM), and deep neural networks (DNN). While ridge regression and SVR (depending on the type of kernel used) can identify a linear relationship between the input features and the target variable, GBMs and DNNs can map more complex nonlinear relationships between the two. We built a convenient and flexible ML pipeline implementing the four methods and consisting of different steps required for constructing these genotype-phenotype maps, such as data pre-processing, feature selection, and model learning using two types of hyperparameter optimization techniques, namely random and Bayesian optimization.


# Installations
```
- git clone https://github.com/SakshiKhaiwal/Genotype-to-phenotype-mapping-in-yeast.git
- cd Genotype-to-phenotype-mapping-in-yeast/pipeline
- conda create -n GenPhen python=3.8
- conda activate GenPhen
- pip install -r requirements.txt
```


# Running the pipeline

</details>

- Input data.
The input dataset to run the pipeline should be in 'CSV' format, where the first column contains the target variable and the rest of the columns are the features. An example of the data file is given in the folder 'data' with the name 'Test_data.csv'

- Pipeline parameters.
To see all the parameters of the pipeline, run:
```
   python main.py --help
```

You will see the following parameters that the user can define.
```

  --data_path DATA_PATH   ### path to the input data
                        

  --data_path_out DATA_PATH_OUT  ### path to the output data
                        

  --data_splitting_criteria {preprocess_data_HOAR,preprocess_data_INHO,preprocess_data_LOCO}
                      ###  How to split the data into training and testing

  --test_fraction float ranging between 0 and 1 ###   fraction of dataset to be used for test set
  --imputation_method {MI,KNN} ###   type of imputation method for missing data

  --clades_data_path CLADES_DATA_PATH ### if splitting criteria is INHO or LOCO, then give the path to the clade path.
                        

  --do_feature_selection True or False  ## Whether to perform feature selection or not
                        

  --feature_selection_strategy {'lasso_selection_grid', 'lasso_selection_random',
                                 'lasso_selection_bayes','high_lasso'} ### Choices of feature selection strategy
                       
  --model_type {'BayesHypOPt_Ridge_regression', 'BayesHypOPt_Elanet_regression',
                                 'BayesHypOPt_GBM_regression','BayesHypOPt_HistGBM_regression',
                                 'BayesHypOPt_SVR_regression','BayesHypOPt_NN_regression',
                                 'RandHypOPt_Ridge_regression','RandHypOPt_Elanet_regression',
                                 'RandHypOPt_GBM_regression','RandHypOPt_SVR_regression',
                                 'RandHypOPt_NN_regression} ### Model to be used for training.

--n_iterations N_ITERATIONS ### Number of iterations for the model
--cross_val CROSS_VAL  ### Number of the cross-validation folds
--num_jobs NUM_JOBS  ### Number of computational threads to use, -1 to use all.


                    


```
- Train the model for the test data. 
To train the model with the default parameters, run the following command:
```
   python main.py --data_path=../data/Test_data.csv --data_path_out=../Output_results 
```


- Results.
The output path should contain two JSON files with the suffix '_prediction_accuracy.json' and '_additional_information.json'. Each file contains a dictionary object, '_prediction_accuracy.json': 'Test r2 score', 'Train r2 score', 'MSE', 'CV mean score', 'CV std', 'Test pears value', 'Train pears value', 'Training time' and the '_additional_information.json': 'y_train_predicted', 'y_test_predicted', and 'Features importance scores'.


# Benchmarking 
To benchmark over different types of genetic features and multiple phenotypes for all models, run the Benchmark.py script using the following command:

```
python Benchmark.py --data_path=INPUT_DATA_PATH --data_path_out=OUTPUT_DATA_PATH

```
In the 'INPUT_DATA_PATH', the genetic matrices are added with the suffix 'gen.csv' and the phenotype matrix with the suffix 'phen.csv'. 


# Dataset.
All the datasets used in the project are provided in the referenced paper.

