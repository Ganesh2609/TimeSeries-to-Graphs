# TimeSeries-to-Graphs: EEG Classification Using Graph Theory

A novel approach to EEG signal classification that transforms time series data into graph representations for distinguishing between alcoholic and control subjects using machine learning techniques.

## Overview

This project implements an innovative methodology for analyzing EEG (Electroencephalogram) signals by converting time series data into graph structures and extracting meaningful features for classification. The approach leverages visibility graph theory and graph-based feature extraction to identify patterns that distinguish between alcoholic and control subjects.

## Methodology

### 1. Data Preprocessing
- **EEG Data Loading**: Processes raw EEG data files from multiple sensors across different subjects
- **Sensor-wise Organization**: Organizes data by 63 EEG sensors including FP1, FP2, F7, F8, AF1, AF2, FZ, FC5, CP3, etc.
- **Class Separation**: Separates data into alcoholic and control groups based on subject classification

### 2. Graph Construction
The project employs **Natural Visibility Graph (NVG)** algorithm to transform time series into graph representations:
- Each time point becomes a node in the graph
- Edges connect nodes based on visibility criteria
- Graph topology captures temporal dependencies and signal characteristics

### 3. Feature Extraction
Comprehensive graph-based features are extracted from each visibility graph:
- **Centrality Measures**: Degree, betweenness, closeness centrality, and PageRank
- **Structural Properties**: Clustering coefficient, average path length, assortativity
- **Graph Complexity**: Graph Index Complexity (GIC), entropy measures
- **Network Topology**: Maximum clique size, minimum cut size, vertex coloring number
- **Efficiency Metrics**: Global efficiency and diameter

### 4. Classification Models
Multiple machine learning algorithms are implemented and compared:
- **Support Vector Machine (SVM)** - Linear kernel
- **K-Nearest Neighbors (KNN)** - Optimized with k=22
- **Random Forest** - Ensemble method with 100 estimators
- **Logistic Regression** - Linear probabilistic classifier
- **Naive Bayes** - Gaussian probabilistic classifier

## Project Structure

```
TimeSeries-to-Graphs/
├── main.py                    # Main processing pipeline
├── utility.py                 # Graph construction and feature extraction
├── classification.ipynb       # ML model implementation and evaluation
├── classification_only.ipynb  # Data loading and preprocessing
├── lda_data_train.csv         # Training dataset with extracted features
├── lda_data_test.csv          # Testing dataset with extracted features
└── eeg_data_test/             # Raw EEG data files
```

## Key Features

### Graph Construction (`utility.py`)
- **Visibility Graph Generation**: Implements natural visibility graph algorithm using ts2vg library
- **Feature Computation**: Extracts 16 comprehensive graph-based features
- **Sensor Processing**: Handles data from 63 EEG sensors simultaneously

### Classification Pipeline (`classification.ipynb`)
- **Model Training**: Implements 5 different classification algorithms
- **Performance Evaluation**: Accuracy assessment on test dataset
- **Feature Selection**: Uses optimal feature subset (FCZ, PO8, TP7, FC5, CP3)

### Data Processing (`main.py`)
- **Batch Processing**: Processes multiple EEG files efficiently with progress tracking
- **JSON Export**: Saves processed graph features for reproducible analysis
- **Modular Design**: Separates data loading, graph construction, and feature extraction

## Results

The classification models achieved the following accuracies on the test dataset:

| Algorithm | Accuracy |
|-----------|----------|
| **SVM (Linear)** | **80.0%** |
| KNN (k=22) | 79.7% |
| Logistic Regression | 79.3% |
| Random Forest | 77.7% |
| Naive Bayes | 77.2% |

The SVM with linear kernel demonstrated the best performance, successfully distinguishing between alcoholic and control subjects with 80% accuracy.

## Technical Implementation

### Dependencies
- **NetworkX**: Graph construction and analysis
- **ts2vg**: Visibility graph generation
- **scikit-learn**: Machine learning algorithms
- **pandas/numpy**: Data manipulation and numerical computing
- **tqdm**: Progress tracking for batch processing

### Graph Features
The system extracts 16 key features from each visibility graph:
1. Degree centrality
2. Betweenness centrality  
3. Closeness centrality
4. PageRank values
5. Degree distribution
6. Average path length
7. Assortativity coefficient
8. Graph diameter
9. Number of edges
10. Clustering coefficient
11. Global efficiency
12. Graph Index Complexity
13. Maximum clique size
14. Minimum cut size
15. Vertex coloring number
16. Shannon entropy

## Usage

1. **Data Processing**: Run `main.py` to convert EEG time series to graph features
2. **Classification**: Execute `classification.ipynb` to train and evaluate ML models
3. **Feature Analysis**: Use extracted features in `lda_data_train.csv` and `lda_data_test.csv`

## Applications

- **Medical Diagnosis**: Automated detection of alcohol-related neurological changes
- **Neuroscience Research**: Understanding brain connectivity patterns
- **Signal Processing**: Novel approach to time series classification
- **Healthcare Technology**: Non-invasive assessment of neurological conditions

## Innovation

This project introduces a novel paradigm for EEG analysis by:
- Transforming temporal signals into spatial graph representations
- Capturing complex signal dependencies through graph topology
- Enabling interpretable feature extraction from brain signals
- Demonstrating superior classification performance compared to traditional methods