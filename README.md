# Fe3O4 GMR-SVM Machine Learning

## Overview

This repository presents a comprehensive machine learning framework for the characterization and quantitative analysis of iron oxide (Fe₃O₄) nanoparticle suspensions utilizing Giant Magnetoresistance (GMR) sensor technology. The project was developed within the MoRA research initiative supported by The AIR Funds and integrates both classification and regression methodologies to establish robust models for magnetic property analysis and concentration prediction.

## Scientific Background

### Giant Magnetoresistance (GMR) Sensors

Giant Magnetoresistance is a quantum mechanical phenomenon wherein the electrical resistance of a conductor exhibits significant variation in response to applied magnetic fields. GMR-based sensors demonstrate exceptional sensitivity to magnetic field variations (typically in the millitesla range) with enhanced signal-to-noise ratios compared to conventional magnetoresistive technologies. These characteristics render GMR sensors particularly advantageous for non-invasive, real-time monitoring of magnetic nanoparticles in suspension.

### Fe₃O₄ Nanoparticles

Iron oxide nanoparticles (Fe₃O₄) are ferrimagnetic materials that exhibit pronounced magnetic properties at room temperature. Their biocompatibility and tunable magnetic characteristics make them invaluable in biomedical applications, including magnetic hyperthermia, drug delivery systems, and biosensing applications. This investigation leverages the distinctive magnetic signature of Fe₃O₄ suspensions to enable concentration-dependent quantification through machine learning methodologies.

## Project Structure

```
Fe3O4-GMR-SVM-Machine-Learning/
├── main/
│   ├── train_classification.py          # Classification model training pipeline
│   ├── train_regression.py              # Regression model training pipeline
│   ├── inference_SVR.py                 # Real-time inference interface with GUI (Publisher)
│   └── inference_subscriber.py          # MQTT subscriber for real-time data reception and concentration prediction
├── models/                              # Trained model storage
│   ├── classification/
│   │   ├── SVM/
│   │   ├── KNN/
│   │   └── RandomForest/
│   └── regression/
│       ├── SVR/
│       ├── KNN/
│       └── RandomForest/
├── output_results/                      # Comprehensive results and visualizations
│   ├── classification/
│   │   ├── json/
│   │   ├── excel/
│   │   └── plots/
│   └── regression/
│       ├── json/
│       ├── excel/
│       └── plots/
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Methodology

### Data Acquisition and Preprocessing

Input data consists of magnetoresistance measurements (ΔB in millitesla) acquired from processed sensor outputs across multiple iterations. The dataset encompasses Fe₃O₄ suspensions with varying concentrations (5, 10, 20, 30, 40, 50 mg/mL), enabling both discrete classification and continuous regression analysis.

### Classification Framework

The classification module (`train_classification.py`) implements three distinct algorithms:

- **Support Vector Classification (SVC)**: Non-linear classification with radial basis function kernel optimization
- **k-Nearest Neighbors (KNN)**: Instance-based learning with Euclidean distance metric
- **Random Forest Classifier**: Ensemble method leveraging multiple decision trees

**Target Variable**: Discrete concentration classes (mg/mL)

**Output Metrics**:
- Accuracy, Precision, Recall, and F1-Score
- Confusion matrices and classification reports
- Cross-validation scores (5-fold stratified)
- ROC-AUC analysis

### Regression Framework

The regression module (`train_regression.py`) implements three regression algorithms:

- **Support Vector Regression (SVR)**: Non-linear regression with kernel methods
- **k-Nearest Neighbors Regressor**: Distance-weighted regression
- **Random Forest Regressor**: Ensemble regression approach

**Target Variable**: Continuous concentration (mg/mL)

**Output Metrics**:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score and Mean Absolute Percentage Error (MAPE)
- Cross-validation evaluation (k-fold)

### Feature Engineering and Preprocessing

All models employ a standardized preprocessing pipeline:

1. **Feature Standardization**: StandardScaler normalization to zero mean and unit variance
2. **Train-Test Split**: 80-20 stratified division for robust generalization assessment
3. **Cross-Validation**: k-fold cross-validation (k=5) for unbiased performance estimation

## Usage

### Prerequisites

```bash
python >= 3.8
numpy
pandas
scikit-learn >= 1.0
matplotlib
seaborn
openpyxl
joblib
paho-mqtt
tkinter (usually included with Python)
pyserial
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Classification Models

```bash
python main/train_classification.py
```

This script performs the following operations:
- Loads processed data from the Excel data source
- Trains SVC, KNN, and Random Forest classifiers
- Generates comprehensive performance metrics (JSON and Excel formats)
- Creates visualizations including confusion matrices and performance comparisons
- Exports trained models for subsequent inference tasks

**Output Files**:
- `output_results/classification/json/classification_metrics.json`
- `output_results/classification/excel/` - Detailed performance tables
- `output_results/classification/plots/` - Performance visualizations

### Training Regression Models

```bash
python main/train_regression.py
```

This script executes the following workflow:
- Loads processed sensor data
- Trains SVR, KNN Regressor, and Random Forest Regressor models
- Computes regression performance metrics
- Generates prediction vs. actual plots and residual analysis
- Exports all trained models with serialization via joblib

**Output Files**:
- `output_results/regression/json/regression_metrics.json`
- `output_results/regression/excel/` - Regression statistics and predictions
- `output_results/regression/plots/` - Model predictions and performance visualizations

### Real-Time Inference Interface (Publisher)

```bash
python main/inference_SVR.py
```

This module provides a graphical user interface (GUI) for real-time concentration prediction as a publisher:

**Functionality**:
- Serial communication with GMR sensor hardware (cross-platform: Windows, Linux, macOS)
- Real-time data visualization with animated plots
- Automatic magnetic field (B) conversion from sensor voltage readings
- Calibration offset: ΔB = 5.3381V - 4.2983
- SVR model-based concentration prediction with manual model selection via dialog box
- MQTT publisher for networked data distribution
- Comprehensive logging and data export capabilities

**Hardware Requirements**:
- GMR sensor with serial interface (9600 baud)
- MQTT broker for message distribution (optional, localhost:1883 default)

**Configuration**:
- Windows: COM3 (default)
- Linux: /dev/ttyUSB0 (default)
- macOS: /dev/tty.usbserial-0001 (default)

### Real-Time Inference Interface (Subscriber)

```bash
python main/inference_subscriber.py
```

This module provides a graphical user interface (GUI) for real-time data reception and concentration prediction as a subscriber:

**Functionality**:
- MQTT subscriber for receiving sensor data from publisher
- Real-time data visualization with animated plots
- Automatic concentration prediction using loaded SVR model
- Display of detected concentration and nearest class
- Comprehensive logging and data export capabilities (including concentration data)
- Statistical analysis of received data

**Requirements**:
- MQTT broker connection (localhost:1883 default)
- Compatible SVR model file (.pkl) for concentration prediction

## Model Performance Summary

Upon successful training execution, comprehensive metrics are generated for all models. The exported JSON and Excel files contain:

### Classification Metrics
- Per-class precision, recall, and F1-score
- Overall accuracy and weighted averages
- Cross-validation mean and standard deviation
- Detailed classification reports per model

### Regression Metrics
- Prediction errors (MAE, MSE, RMSE)
- Goodness-of-fit measures (R², MAPE)
- Cross-validation performance statistics
- Per-sample prediction accuracies

## Research Applications

This framework is applicable to:

1. **Magnetic Nanoparticle Characterization**: Quantitative analysis of Fe₃O₄ concentration in suspension
2. **Biosensing Applications**: Detection and quantification of magnetic biomarkers
3. **Quality Control**: Process monitoring in nanoparticle synthesis and purification
4. **Clinical Diagnostics**: Point-of-care testing with GMR-based sensors
5. **Environmental Monitoring**: Detection of magnetic contaminants in aqueous systems

## Technical Implementation Details

### Feature Input
- **Primary Feature**: Magnetoresistance change (ΔB in millitesla)
- **Data Source**: Processed sensor acquisition sheets from multi-iteration measurements
- **Preprocessing**: Standardized normalization across all training instances

### Model Training Parameters
- **Train-Test Ratio**: 80:20 stratified split
- **Cross-Validation Strategy**: k-fold (k=5) for classification; standard k-fold for regression
- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Random State**: Fixed for reproducibility across executions

### Output Formats
- **Metrics**: JSON (machine-readable) and Excel (human-readable)
- **Plots**: PNG format with high-resolution DPI settings
- **Models**: Binary serialization via joblib for efficient inference

## Results Visualization

Generated visualizations include:

**Classification Outputs**:
- Confusion matrices for each algorithm
- Comparative performance bar charts
- ROC curves and AUC metrics
- Cross-validation score distributions

**Regression Outputs**:
- Prediction vs. actual scatter plots
- Residual analysis plots
- Error distribution histograms
- Model comparison line plots

## Dependencies and Licensing

### Core Libraries
- **scikit-learn**: Machine learning algorithms and metrics
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **openpyxl**: Excel workbook generation
- **joblib**: Model serialization
- **paho-mqtt**: MQTT client for distributed systems
- **tkinter**: GUI framework (standard with Python)
- **pyserial**: Serial communication for hardware interface

All dependencies are open-source and available through PyPI.

## Future Development Directions

1. **Deep Learning Integration**: Implementation of convolutional and recurrent neural networks for temporal data analysis
2. **Hyperparameter Optimization**: Bayesian optimization and grid search methodologies
3. **Ensemble Methods**: Stacking and voting classifiers for improved robustness
4. **Time-Series Analysis**: Temporal correlation assessment and autoregressive models
5. **Hardware Integration**: Firmware optimization and real-time data streaming enhancements

## Contact and Support

For inquiries regarding methodology, implementation details, or research collaboration opportunities, please contact the research team.

---

**Last Updated**: May 12, 2026  
**Version**: 1.1
