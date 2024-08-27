# Genotype to phenotype prediction of natural variation in **S. cerevisiae**.

In this project, we investigate the associations between the phenotypes and genotypes in a natural yeast population. We have gathered >200 phenotypes from various studies for 1011 **cerevisiae** strains. To construct meaningful models that can provide predictions of the phenotype in a novel population based on its genotypes, we explored several linear and nonlinear machine learning (ML) methods. Since we are interested in the prediction of quantitative phenotypes, we utilized regression-based methods, such as ridge regression, support vector regression (SVR), gradient-boosted machines (GBM), and deep neural networks (DNN). While ridge regression and SVR (depending on the type of kernel used) can identify a linear relationship between the input features and the target variable, GBMs and DNNs can map more complex nonlinear relationships between the two. We built a convenient and flexible ML pipeline implementing the four methods and consisting of different steps required for constructing these genotype-phenotype maps, such as data pre-processing, feature selection, and model learning using two types of hyperparameter optimization techniques, namely random and Bayesian optimization.


# Installations
```
- git clone git@github.com:SakshiKhaiwal/GenPhen.git
- cd pipeline
- conda create -n GenPhen python=3.8
- conda install --file requirements.txt
```


# Running the pipeline

</details>

- Input data.
The input dataset to run the pipeline should be in 'CSV' format, where the first column contains the target variable and the rest of the columns are the features. An example of the data file is given in the folder 'data' with the name 'Test_data.csv'

- Pipeline parameters.
To see all the parameters of the pipeline, run:
```
   -python main.py --help
```

```

  --data_path DATA_PATH  ## path to the input data
                      
  --data_path_out DATA_PATH_OUT   ## path to the output data
                      
  --data_splitting_criteria {preprocess_data_randomsplit,preprocess_data_cladesplit} ## splitting crietria

  --clades_data_path CLADES_DATA_PATH  ## splitting criteria == cladesplit, then provide path to clades data
                      
  --do_feature_selection DO_FEATURE_SELECTION   ##  apply feature selection
                    
  --feature_selection_strategy {lasso_selection_grid,lasso_selection_random,lasso_selection_bayes,high_lasso} ## choice of feature selection strategy
                      
  --model_type {BayesHypOPt_Ridge_regression,BayesHypOPt_Elanet_regression,BayesHypOPt_GBM_regression,BayesHypOPt_SVR_regression,BayesHypOPt_NN_regression,RandHypOPt_Ridge_regression,RandHypOPt_Elanet_regression,RandHypOPt_GBM_regression,RandHypOPt_SVR_regression,RandHypOPt_NN_regression} ##   used model
                    


```
- Train the model. 
To train the model with the default parameters, run the following command:
```
   -python main.py  --data_path=INPUT_DATA --data_path_out=OUTPUT_DATA_PATH
```


- Results.
The output path should contain two JSON files with the suffix '_prediction_accuracy.json' and '_additional_information.json'. Each file contains a dictionary object, '_prediction_accuracy.json': 'Test r2 score', 'Train r2 score', 'Test pears value': Test_pears_val, 'Train pears value', 'Training time' and the '_additional_information.json': 'y_train_predicted', 'y_test_predicted', 'training_strains', 'testing_strains' and 'Features importance scores'.

# Dataset.
All the datasets used in the project are provided with the ref. paper.

