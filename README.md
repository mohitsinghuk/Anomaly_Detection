# Anomaly_Detection
## Overview
This project aims to develop a machine learning framework for anomaly detection within IoT (Internet of Things) networks, utilizing edge computing and federated learning techniques. By focusing on multi-variable time-series data from IoT devices, the framework detects deviations in normal behavior in real time, ensuring data privacy, reducing latency, and optimizing bandwidth usage.


## Project Structure

Data Collection & Preprocessing:

Gathered IoT datasets from publicly accessible sources (e.g., Harvard Dataverse).
Cleaned, normalized, and balanced the dataset to prepare for model training and ensure quality analysis.
Model Development:

Built an Autoencoder-based neural network to detect anomalies in time-series data.
Integrated Federated Learning to allow decentralized model training across edge devices, keeping raw data local and private.
Optimized the model for edge devices by applying quantization and model compression techniques.
Evaluation Metrics:

Evaluated model performance based on accuracy, precision, recall, F1-score, and latency.
Tested scalability and efficiency of the framework on real-world IoT data.
Project Highlights:

Enhanced real-time anomaly detection while preserving data privacy.
Efficient deployment suitable for large-scale IoT networks with minimal resource consumption.


## Objectives

Real-Time Anomaly Detection: Enable edge devices to detect anomalies locally, reducing reliance on centralized servers and ensuring quick responses.
Data Privacy and Security: Leverage federated learning to maintain privacy, only sharing model updates between devices rather than raw data.
Scalability and Efficiency: Develop a model capable of handling large-scale IoT data while maintaining high detection accuracy and computational efficiency.


## Tools and Technologies

Languages: Python
Libraries & Frameworks:
PyTorch: For developing and training neural networks, particularly autoencoders.
PySyft: For privacy-preserving machine learning and federated learning implementation.

Data Processing:
NumPy & Pandas: For data manipulation.
Min-Max Normalization: To ensure uniform data scaling.
Visualization: Matplotlib and Seaborn for data exploration and model performance visualization.
Deployment Tools: Jupyter Notebooks for interactive model development and documentation.


## Dataset

Source: The Shuttle dataset from Harvard Dataverse, containing both normal and anomalous data points, simulating real-world IoT conditions.
Key Features: Attributes include flow rate, pressure level, sensor temperature, and equipment status.
Challenge: Highly imbalanced dataset with only 1.89% anomalies, addressed through stratified sampling and advanced evaluation metrics like precision and recall.


## Methodology

1. CRISP-DM Framework:
Adopted the Cross-Industry Standard Process for Data Mining for iterative model development.
Phases include business understanding, data understanding, data preparation, modeling, evaluation, and deployment.
2. Model Training:
Autoencoders learn compressed representations of normal data, flagging anomalies based on reconstruction errors.
Federated Learning enables multiple edge devices to train collaboratively without transferring raw data to a central server, ensuring privacy and reducing bandwidth usage.
3. Evaluation & Testing:
Metrics: Accuracy, precision, recall, F1-score, and latency to ensure effective anomaly detection.
Visualization: Confusion matrices, ROC curves, and reconstruction error plots to interpret model performance.


## Key Project Contributions
Enhanced Detection Accuracy: Achieved robust detection of point, contextual, and collective anomalies in IoT data.
Optimized for Edge Devices: Model scalability and efficiency were optimized for deployment on edge computing devices.
Privacy-Preserving: Federated learning ensured data privacy by preventing raw data transfer and using secure model updates.


## How to Run the Project
Clone Repository: Download project files.
Install Dependencies: Use pip install -r requirements.txt to set up the Python environment.
Data Preprocessing: Use provided scripts to preprocess and normalize the IoT dataset.
Train the Model: Run the training script to build and evaluate the anomaly detection model.
Evaluate & Test: Use testing scripts to assess model performance on unseen data.
Deployment: Implement the model on edge devices for real-time anomaly detection.


## Future Work
Model Improvements: Experiment with advanced architectures like LSTMs and GRUs for improved anomaly detection in complex sequences.
Adaptive Thresholding: Implement dynamic thresholding for anomaly detection to handle varying data patterns across different edge devices.
Real-World Deployment: Test the model in various real-world IoT environments, such as smart cities, autonomous vehicles, and industrial systems.


## Ethical Considerations
Data Privacy: Ensured ethical handling of sensitive data through federated learning, with all raw data remaining localized.
Transparency: Model updates shared securely and transparently, adhering to data privacy regulations.
