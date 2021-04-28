# TEST 2
import unittest
import torch
import os


from context import src
from src.regressor import Regressor



path = os.path.join("..", "models", "reg_nn.pth")
reg = Regressor()
reg.load_state_dict(torch.load(path,  map_location=torch.device('cpu')))
reg.eval()



class BasicTestSuite2(unittest.TestCase):
    """Basic test cases."""


    def test_model_hidden_size(self):
    	assert reg.layer2.out_features == 25, "Hidden size should be 25"


    def test_model_input_size(self):
        assert reg.layer1.out_features == 50, "Input size should be 50"

if __name__ == '__main__':
    unittest.main()
