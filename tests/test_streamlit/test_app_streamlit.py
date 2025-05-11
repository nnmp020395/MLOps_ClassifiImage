import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image
import requests
from prometheus_client import REGISTRY, Counter

class TestStreamlitApp(unittest.TestCase):

    @patch("requests.post")
    def test_api_prediction_success(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prediction": "Dandelion"}
        mock_post.return_value = mock_response

        # Simulate an image file upload
        image = Image.new("RGB", (100, 100), color="green")
        img_bytes = BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Call the API manually
        response = requests.post(
            url="http://fastapi-api:8000/predict",
            files={"file": img_bytes.getvalue()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "Dandelion")
        mock_post.assert_called_once()

    def test_metrics_counter_increment(self):

        # Setup or retrieve the counter
        try:
            counter = REGISTRY._names_to_collectors["test_counter"]
        except KeyError:
            counter = Counter("test_counter", "Test counter increment")

        before = counter._value.get()
        counter.inc()
        after = counter._value.get()

        self.assertEqual(after, before + 1)

