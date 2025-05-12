import sys
import unittest
from unittest.mock import MagicMock, patch
import io
import torch
import torch.nn as nn

parent_folder = "/src"
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

from src.model import DinoClassifier, load_model

class TestDinoClassifier(unittest.TestCase):

    def setUp(self):
        # Create a mock backbone
        self.backbone = MagicMock(spec=nn.Module)
        self.backbone.embed_dim = 384  # Example embedding dimension for DINOv2

        # Create an instance of DinoClassifier
        self.model = DinoClassifier(self.backbone, num_classes=2)

    def test_forward(self):
        # Create a mock input tensor
        input_tensor = torch.randn(1, 3, 224, 224)

        # Mock the backbone's forward method
        self.backbone.return_value = torch.randn(1, self.backbone.embed_dim)

        # Call the forward method
        output = self.model(input_tensor)

        # Check the output shape
        self.assertEqual(output.shape, (1, 2))

class TestLoadModel(unittest.TestCase):

    @patch('your_script.s3fs.S3FileSystem')
    @patch('your_script.torch.load')
    def test_load_model_from_s3(self, mock_torch_load, mock_s3fs):
        # Mock the S3 filesystem and torch.load
        mock_fs = MagicMock()
        mock_s3fs.return_value = mock_fs
        mock_file_content = b"fake_model_content"
        mock_fs.open.return_value.__enter__.return_value.read.return_value = mock_file_content

        # Mock the loaded state dict
        mock_state_dict = {"key": "value"}
        mock_torch_load.return_value = mock_state_dict

        # Call the load_model function with an S3 path
        model = load_model("s3://bucket/model.pth")

        # Check that the model was loaded correctly
        self.assertIsInstance(model, DinoClassifier)
        mock_torch_load.assert_called_once()

    @patch('your_script.torch.load')
    def test_load_model_from_local(self, mock_torch_load):
        # Mock the loaded state dict
        mock_state_dict = {"key": "value"}
        mock_torch_load.return_value = mock_state_dict

        # Call the load_model function with a local path
        model = load_model("local/model.pth")

        # Check that the model was loaded correctly
        self.assertIsInstance(model, DinoClassifier)
        mock_torch_load.assert_called_once()

    @patch('your_script.torch.load')
    def test_load_model_from_buffer(self, mock_torch_load):
        # Mock the loaded state dict
        mock_state_dict = {"key": "value"}
        mock_torch_load.return_value = mock_state_dict

        # Create a BytesIO buffer with fake content
        buffer = io.BytesIO(b"fake_model_content")

        # Call the load_model function with a buffer
        model = load_model(buffer)

        # Check that the model was loaded correctly
        self.assertIsInstance(model, DinoClassifier)
        mock_torch_load.assert_called_once()

if __name__ == "__main__":
    unittest.main()
