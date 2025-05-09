import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

sys.modules['model'] = MagicMock()

from src.inference import predict

class TestPredictFunction(unittest.TestCase):
    @patch("src.inference.Image.open")
    @patch("src.inference.model")
    @patch("src.inference.transform")
    def test_predict_dandelion(self, mock_transform, mock_model, mock_open):
        dummy_image = MagicMock(spec=Image.Image)
        mock_open.return_value = dummy_image

        transformed_tensor = torch.rand(1, 3, 224, 224)
        mock_transform.return_value = transformed_tensor

        dummy_output = torch.tensor([[1.0, 0.0]])
        mock_model.return_value = dummy_output

        result = predict("dummy_path.jpg")

        mock_open.assert_called_once_with("dummy_path.jpg")
        mock_transform.assert_called_once_with(dummy_image)
        mock_model.assert_called_once()
        self.assertEqual(result, "dandelion")

if __name__ == '__main__':
    unittest.main()
