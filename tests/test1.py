# TEST 1
import unittest
import torch
import os
import torch.nn as nn

from context import src
from src.regressor import Regressor




path = os.path.join("..", "models", "reg_nn.pth")
reg = Regressor()
reg.load_state_dict(torch.load(path,  map_location=torch.device('cpu')))
reg.eval()



class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    
    def test_model_parameters(self):
    	assert len(list(reg.named_parameters())) == 6, "Parms should be 6"

    def test_model_output_size(self):
    	assert reg.output.out_features == 1, "Output size should be 1"


if __name__ == '__main__':
    unittest.main()
