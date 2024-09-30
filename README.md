# Hindi OCR Model Deployment using Streamlit

This repository contains the deployment of a Hindi Optical Character Recognition (OCR) model using the [Streamlit](https://streamlit.io/) framework. The model is designed to recognize and classify Hindi text from images and is trained on a complex dataset using transfer learning techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Training](#model-training)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Deployment](#streamlit-deployment)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to deploy a machine learning model for recognizing Hindi text in images. The model was trained using transfer learning and handles multi-class classification for thousands of Hindi words. The goal is to make this OCR tool accessible via a web-based application where users can upload images and get the corresponding Hindi text.

### Features

- Recognizes Hindi text from images.
- Allows users to upload images directly via the web interface.
- Provides real-time inference using the trained OCR model.
- Displays recognized text and accuracy scores.

## Model Training

The model was trained using transfer learning based on a pre-trained model from the `transformers` library. The following steps were performed during training:

- **Data preprocessing**: Images and labels were loaded, preprocessed, and converted to tensors.
- **Model**: Transfer learning from a base OCR model (`stepfun-ai/GOT-OCR2_0`).
- **Optimization**: The model was fine-tuned with a custom dataset containing Hindi characters and words.
- **Loss function**: Categorical Cross-Entropy.
- **Optimizer**: Adam optimizer.
- **Evaluation**: The model was evaluated on test data, which showed an accuracy of 0.00015. Future work is required to improve this accuracy.

## Installation

### Prerequisites

- Python 3.7+
- pip
- Streamlit
- TensorFlow or PyTorch (depending on the model backend)
- Other dependencies listed in `requirements.txt`

### Clone the repository

```bash
git clone https://github.com/your-username/hindi-ocr-streamlit.git
cd hindi-ocr-streamlit


