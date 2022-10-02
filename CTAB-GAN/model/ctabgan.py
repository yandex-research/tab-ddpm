"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 df,
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'},
                 batch_size = 512,
                 class_dim = (256, 256, 256, 256),
                 lr = 2e-4,
                 epochs = 10,
                 device=None):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(lr = lr, epochs = epochs, batch_size = batch_size, class_dim = class_dim, device = device)
        self.raw_df = df
        print(self.raw_df.shape)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        
    def fit(self, no_train=False):
        print("-"*100)
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], 
        mixed = self.data_prep.column_types["mixed"],type=self.problem_type, no_train=no_train)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")
        print("-"*100)


    def generate_samples(self, num_samples, seed=0):
        
        sample = self.synthesizer.sample(num_samples, seed) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
