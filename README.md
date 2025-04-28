# FastAPI Resource Performance Predictor
This project is a FastAPI server that predicts future CPU and RAM usage based on recent performance data.
It loads trained TensorFlow models and scalers to make 6-step ahead forecasts.

## Project Structure
```bash
/AIServerAIServer
    |- main.py
    |- requirements.txt
    |- Dockerfile
    |- docker-compose.yml
    |- README.md
    |- /model
         |- model_cpu.keras
         |- model_ram.keras
         |- scaler_X.pkl
         |- scaler_y.pkl
         |- scaler_X_ram.pkl
         |- scaler_y_ram.pkl

```
## Requirements (If running locally)

- Python 3.10+

- Docker (for containerization)

- TensorFlow

- FastAPI

- Uvicorn

- Pandas

- NumPy

- scikit-learn

- joblib

- python-multipart

### Install requirements manually:
```bash
pip install -r requirements.txt
```
## How to Build and Run with Docker
### 1. Build Docker Image
Open a terminal (Command Prompt or PowerShell), navigate to the project directory, and build the image:
```bash
docker build -t fastapi-predictor .
```
- "-t" fastapi-predictor sets the image name.

- "." means build context is current directory.
### 2. Run Docker Container
Start the container mapping container's port 8000 to your local port 8000:
```bash
docker run -d -p 8000:8000 fastapi-predictor
```
## License
This project is for educational and internal usage.
