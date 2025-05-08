# Medical Image Classification

This repository contains a machine learning model for classifying medical images (X-rays, MRIs, CT scans) to predict whether a patient has **pneumonia** or is **normal**. The model was built using a convolutional neural network (CNN) and trained on a dataset of medical images.

## Problem Statement

The goal of this project is to develop a model that can automatically classify medical images to assist healthcare professionals in diagnosing diseases like pneumonia. The model takes in images, processes them, and outputs a classification: either **normal** or **pneumonia**.

## Dataset

The dataset consists of chest X-ray images that are labeled into two categories:

- **NORMAL**: Images that depict healthy lungs.
- **PNEUMONIA**: Images that depict pneumonia-infected lungs.

The dataset is divided into three parts:

- **Training Set**: Images used to train the model.
- **Validation Set**: Images used to validate the model's performance during training.
- **Test Set**: Images used to evaluate the final model performance.

## System Overview

The system utilizes a Convolutional Neural Network (CNN) architecture to classify medical images. It works by training the model on a labeled dataset and then testing it on unseen data to predict if a chest X-ray belongs to a healthy patient or one with pneumonia.

## Technologies Used

- **Python**: The primary programming language used.
- **TensorFlow/Keras**: Deep learning framework for building and training the model.
- **Matplotlib**: Library used for plotting training progress and results.
- **NumPy**: Used for numerical operations and data handling.
- **OpenCV**: Image processing tool to preprocess images before feeding them to the model.
- **GitHub**: Version control and hosting.

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/raeen-fatima/medical-image-classification.git
   cd medical-image-classification
2. Install the required dependencies:

   pip install -r requirements.txt

3.Download the dataset and place it in the appropriate folder (/data).

    python train.py
    This script will:
    Load the dataset.
    Preprocess the images.
    Train the CNN model.
    Save the trained model to disk for future use.
    Evaluating the Model
    To evaluate the model on test data:

        python evaluate.py
This will display the model's accuracy and other performance metrics.

## Future Enhancements
Integration with Cloud Services: Deploy the model on a cloud platform for easy access.

Additional Classes: Expand the model to classify more types of diseases from medical images.

Real-Time Prediction: Implement real-time predictions using a webcam or mobile app.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements
Special thanks to the dataset providers for making the medical image dataset available.

Thanks to TensorFlow and Keras for providing easy-to-use frameworks for model development.


