# Basic Text Classifier

A binary text classification model built using DistilBERT for toxicity detection in text comments. This implementation is optimized for Apple Silicon GPUs using Metal Performance Shaders (MPS).

## Overview

This project implements a text classifier that categorizes text comments as either "safe" or "unsafe" based on various toxicity parameters. It uses DistilBERT, a lightweight BERT variant, fine-tuned on a Kaggle dataset for toxic comment classification.

## Features

- Binary classification (safe/unsafe)
- Optimized for Apple Silicon GPUs using MPS
- Uses Hugging Face's Transformers library
- Automatic hardware detection (GPU/CPU)
- Built-in logging for monitoring training progress

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy
- scikit-learn
- datasets

## Hardware Support

The classifier automatically detects and utilizes:
- Apple Silicon GPU (MPS) if available
- Falls back to CPU if MPS is not available

## Dataset

The model is trained on a Kaggle dataset that includes multiple toxicity labels:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

These labels are combined into a single binary classification (safe/unsafe).

## Usage

1. Ensure you have the required dependencies installed
2. Place your Kaggle dataset (`train.csv`) in the project directory
3. Run the classifier:
   ```bash
   python run_mps.py
   ```

## Implementation Details

- Uses DistilBERT for text classification
- Implements custom metrics tracking
- Features automatic dataset preprocessing
- Includes logging for monitoring training progress

## License

MIT
