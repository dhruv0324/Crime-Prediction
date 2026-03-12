# Crime Prediction MLOps Pipeline

An end-to-end MLOps pipeline for predicting violent crimes per population using the Communities and Crime dataset. This project demonstrates industry-grade practices including data versioning, experiment tracking, model registry, containerization, CI/CD, and cloud deployment.

## 📋 Project Overview

- **Dataset**: Communities and Crime (1994 instances, 122 predictive features)
- **Task**: Regression (predicting ViolentCrimesPerPop)
- **Goal**: Build a production-ready ML pipeline with full MLOps capabilities

## 🏗️ Architecture

The pipeline includes:
- **Data Versioning**: DVC for data and model versioning
- **Experiment Tracking**: MLflow and Weights & Biases
- **Model Registry**: MLflow Model Registry
- **API Serving**: FastAPI REST API
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Cloud Deployment**: AWS/GCP/Azure ready

## 📁 Project Structure

```
crime-prediction/
├── data/
│   ├── raw/                    # Original data (DVC tracked)
│   ├── processed/              # Processed data (DVC tracked)
│   └── external/               # External data sources
├── src/
│   ├── data/                   # Data processing scripts
│   ├── models/                 # Model training and evaluation
│   ├── features/               # Feature engineering
│   └── api/                    # FastAPI application
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks for EDA
├── scripts/                    # Utility scripts
├── .github/workflows/          # CI/CD pipelines
├── docker/                     # Dockerfiles
├── config/                     # Configuration files
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- Git
- (Optional) Docker for containerization

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd crime-prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

5. **Set up DVC for data versioning**:
   ```bash
   bash scripts/setup_dvc.sh
   ```

6. **Verify installation**:
   ```bash
   python -c "import mlflow, dvc, fastapi; print('Installation successful!')"
   ```

## 📊 Dataset

The Communities and Crime dataset combines:
- 1990 US Census data (socio-economic features)
- 1990 US LEMAS survey (law enforcement features)
- 1995 FBI UCR (crime data)

**Note**: The dataset contains missing values (especially in LEMAS data). This is handled in the preprocessing pipeline.

## 🔧 Configuration

Configuration is managed through YAML files in the `config/` directory:
- `config/config.yaml`: Main project configuration
- `config/experiment_config.yaml`: Experiment parameters (to be created)

Key configuration sections:
- Data paths and split parameters
- Model training parameters
- API serving configuration
- MLOps tool settings (MLflow, W&B, DVC)

## 🧪 Usage

### Data Preprocessing

```bash
python -m src.data.preprocess
```

### Model Training

```bash
python -m src.models.train
```

### Start API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
pytest tests/
```

## 🛠️ Development

### Code Quality

- **Formatting**: `black src/ tests/`
- **Linting**: `flake8 src/ tests/`
- **Type Checking**: `mypy src/`

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes following code style guidelines
3. Write tests for new functionality
4. Run tests and quality checks
5. Submit pull request

## 📈 MLOps Workflow

1. **Data Versioning**: Raw and processed data tracked with DVC
2. **Experiment Tracking**: All experiments logged to MLflow/W&B
3. **Model Registry**: Best models registered and versioned
4. **Automated Testing**: CI/CD pipeline runs tests on every commit
5. **Model Deployment**: Automated deployment to cloud platform
6. **Monitoring**: Model performance and API metrics tracked

## 🌐 Deployment

### Docker

```bash
docker build -t crime-prediction:latest -f docker/Dockerfile .
docker run -p 8000:8000 crime-prediction:latest
```

### Cloud Deployment

The project is configured for deployment on:
- **AWS**: ECS, SageMaker, Lambda
- **GCP**: Cloud Run, AI Platform
- **Azure**: Container Instances, ML Service

---

**Note**: This is a learning project demonstrating MLOps best practices. For production use, ensure proper security, monitoring, and compliance measures are in place.
