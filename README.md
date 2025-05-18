# ML Assets Prediction

A modular Python application for collecting, preprocessing, and predicting asset prices using machine learning. This project is designed to integrate with a larger trading engine. It features support for both local and remote data providers, model training, and prediction workflows.

> **Note:**  
> Throughout this documentation and codebase, the term **Assets** is used as a generic term to describe both **stocks** and **cryptocurrencies**.

## Architecture Overview

> **Note:**  
> Work in progress. The architecture is still being refined.

```
[Data Source] --> [HistoryDataProvider] --> [PreProcessor] --> [Model] --> [Prediction Output]
```

## Features

- Fetch and update historical stock data from various sources.
- Preprocess and clean financial time series data.
- Train and evaluate ML models for stock prediction.
- Save/load models locally or from S3.
- Easily extensible for new data sources or models.

## Supported Providers

- **Local CSV**: Load historical data from local CSV files.

## Supported ML Models

- **Random Forest**: A robust ensemble learning method for regression tasks.

## Folder Structure

```
ml-stocks-prediction/
├── src/
│   ├── providers/
│   │   ├── clients/
│   │   ├── preprocessors/
│   │   └── ...
│   ├── training/
│   └── ...
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Development

### Prerequisites

- Python 3.8+
- pip
- (Optional) virtualenv or conda

### Installation

```bash
git clone https://github.com/toniton/ml-assets-prediction.git
cd ml-assets-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Contributing 🤝 
We welcome contributions from the community! Whether it's bug fixes, new features, or RFCs (Request for Comments), your input is highly valued. Here's how to get started:

### How to Contribute

1. Fork the repository.
2. Create a branch for your feature or fix:
    ```bash
    git checkout -b feat/your-feature-name
    ```
3. Make your changes, write tests if applicable, and ensure the code passes linting and CI checks.

4. Submit a Pull Request (PR) with a clear explanation of your changes.

> **Note:**  
> For larger changes or proposals, open an Issue using the RFC template before starting implementation.