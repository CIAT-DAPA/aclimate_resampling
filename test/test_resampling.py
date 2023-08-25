
import unittest
import pandas as pd
import os
import sys
import numpy as np

test_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(test_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from src.aclimate_resampling.resampling import Resampling


class TestResampling(unittest.TestCase):
    def setUp(self):
        # Define sample input data for testing
        self.country = "ETHIOPIA"
        self.path = test_dir
        self.path_inputs = os.path.join(self.path,self.country,"inputs")
        self.path_inputs_prediccion = os.path.join(self.path_inputs,"prediccionClimatica")
        self.path_inputs_daily = os.path.join(self.path_inputs_prediccion,"dailyData")
        self.path_outputs = os.path.join(self.path,self.country,"outputs")
        self.path_outputs_pred = os.path.join(self.path_outputs,"prediccionClimatica")
        self.path_outputs_prob = os.path.join(self.path_outputs_pred,"probForecast")
        self.year_forecast = 2023
        self.npartitions = 10 
        self.station = "5e91e1c214daf81260ebba59"
        self.out_st =  os.path.join(self.path_outputs_pred, self.station)
        self.out_st_sum =  os.path.join(self.path_outputs_pred, "summary")


    def test_output_folder_creation(self):

        print(self.out_st)
        print(self.out_st_sum)
        self.assertTrue(os.path.exists(self.out_st))
        self.assertTrue(os.path.exists(self.out_st_sum))

    def test_exists_files(self):

        #clim_files = os.path.join(self.path_inputs_daily ,"f{self.station}.csv")
        #prob_files = os.path.join(self.path_outputs_prob , "probabilities.csv")
        scenary_file = os.path.join(self.out_st ,self.station + "_escenario_1.csv")
        sum_file = os.path.join(self.out_st_sum, self.station + "_escenario_max.csv")
        print(scenary_file)
        print(sum_file)
        self.assertTrue(os.path.exists(scenary_file))
        self.assertTrue(os.path.exists(sum_file))




if __name__ == '__main__':
    unittest.main()

























