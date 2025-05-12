import unittest
from unittest.mock import MagicMock, patch
import io
import sys
import torch
from PIL import Image
from torch.utils.data import Subset
from torchvision.transforms import transforms

parent_folder = "/src"
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

from src.train import S3ImageFolder, stratified_split

class TestS3ImageFolder(unittest.TestCase):

    def setUp(self):
        # Mock the S3 filesystem
        self.mock_fs = MagicMock()
        self.mock_fs.ls.return_value = [
            "s3://image-dandelion-grass/raw/dandelion/image1.jpg",
            "s3://image-dandelion-grass/raw/dandelion/image2.jpg",
            "s3://image-dandelion-grass/raw/grass/image1.jpg"
        ]

        # Mock the file content
        self.mock_file_content = b"fake_image_content"

        def open_side_effect(path, mode):
            return io.BytesIO(self.mock_file_content)

        self.mock_fs.open.side_effect = open_side_effect

        # Define the root paths and transform
        self.root_paths = [
            "s3://image-dandelion-grass/raw/dandelion",
            "s3://image-dandelion-grass/raw/grass"
        ]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Create an instance of S3ImageFolder
        self.dataset = S3ImageFolder(self.root_paths, transform=self.transform, fs=self.mock_fs)

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_getitem(self):
        image, label = self.dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(label, 0)

class TestStratifiedSplit(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset
        self.mock_dataset = MagicMock()
        self.mock_dataset.samples = [
            ("s3://image-dandelion-grass/raw/dandelion/image1.jpg", 0),
            ("s3://image-dandelion-grass/raw/dandelion/image2.jpg", 0),
            ("s3://image-dandelion-grass/raw/grass/image1.jpg", 1),
            ("s3://image-dandelion-grass/raw/grass/image2.jpg", 1)
        ]

    def test_stratified_split(self):
        train_data, val_data = stratified_split(self.mock_dataset, split_ratio=0.5)
        self.assertEqual(len(train_data), 2)
        self.assertEqual(len(val_data), 2)

if __name__ == "__main__":
    unittest.main()
