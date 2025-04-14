# MLOps_ClassifiImage

This page contains the MLOps project for classifying images from the 'dandelion' and 'grass' groups. We are developing a web app that allows users to use our deployment.

# How setup environment
1. create virtual environment
```python
python -m venv .venv
```

2. activate virtual environment
- on linux/MacOs
```bash
source .venv/bin/activate
```

-   on windows
``` powershell
.venv\Scripts\activate.bat
```

3. install dependancies
```python
python -m pip install -r requirements.txt
```

# Run Docker
For the 1st time
```bash
docker compose up -d -build
```
otherwise
```bash
docker compose up
```
# Run webapp
After running the docker compose, click on `streamlit` image to run the webapp brownser.
