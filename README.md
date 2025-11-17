# MLOps Playground
An MLOps project demonstrating CI/CD pipelines, automated model training, and inference workflows for machine learning systems.

## Overview
This project showcases production-grade ML engineering practices using the California Housing dataset for house price prediction. It demonstrates the separation of code validation, model training, and inference, reflecting real-world ML system architecture.

**Key Technologies:** Python, scikit-learn, GitHub Actions, pytest, black, flake8

## Architecture
CI Pipeline (Automated)
├── Code quality checks (flake8, black)
├── Unit tests
└── Fast feedback (< 3 minutes)

Training Pipeline (On-Demand)
├── Train RandomForest model
├── Upload to Model Registry (GitHub Artifacts)
└── Model validation

Inference Pipeline (On-Demand)
├── Download latest model from Registry
├── Load model and run predictions
└── Return results

## Project Structure
mlops-playground/
├── .github/workflows/
│   ├── ci.yml              # Code quality + tests
│   ├── train.yml           # Model training
│   └── inference.yml       # Inference service
├── src/
│   ├── train.py            # Training script
│   ├── predict.py          # Local prediction
│   └── inference.py        # GitHub Actions inference
├── tests/
│   ├── test_train.py       # Training tests
│   └── test_predict.py     # Prediction tests
├── requirements.txt        # Production dependencies
└── requirements-dev.txt    # Development dependencies