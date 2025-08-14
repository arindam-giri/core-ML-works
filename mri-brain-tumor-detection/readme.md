# Brain Tumor Detection with CNN

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for brain tumor detection using MRI images. The model classifies images into four classes, achieving a test accuracy of **98.47%** on a dataset of 5,717 training and 1,316 testing images. The architecture combines standard convolutional layers with residual blocks for robust feature extraction, optimized for small medical imaging datasets.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements a CNN to classify MRI brain images into four tumor-related classes. Key features include:
- **Hybrid Architecture**: Combines standard conv layers with residual blocks to enhance feature learning while preventing vanishing gradients.
- **Data Augmentation**: Random flips and rotations to improve generalization on a small dataset.
- **Class Weighting**: Addresses class imbalance to ensure balanced performance across classes.
- **Training Features**: Early stopping, learning rate scheduling, and mixed-precision training for efficiency on Apple MPS or CPU devices.
- **Performance**: Achieves 98.47% test accuracy, with balanced precision, recall, and F1 scores (~0.98).

## Dataset
The model is trained on a private dataset of MRI brain images:
- **Training Set**: 5,717 images
- **Testing Set**: 1,316 images
- **Classes**: 4 (e.g., different tumor types or no tumor; labeled Class 0-3)
- **Directory Structure**:
  ```
  /path/to/dataset/
  ├── Training/
  │   ├── Class0/
  │   ├── Class1/
  │   ├── Class2/
  │   ├── Class3/
  ├── Testing/
  │   ├── Class0/
  │   ├── Class1/
  │   ├── Class2/
  │   ├── Class3/
  ```
**Note**: The dataset is not included in this repository. Update `base_path` in the code to point to your dataset.

## Requirements
- Python 3.8+
- PyTorch 2.0+ (with MPS support for Apple Silicon)
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn scikit-learn
   ```
3. Update the `base_path` in the code to your dataset directory:
   ```python
   base_path = "/path/to/your/dataset"
   ```

## Model Architecture
The CNN combines standard convolutional layers with residual blocks to balance depth and training stability. Input images are resized to 224x224, and the model outputs probabilities for 4 classes.

### Architecture Diagram
```mermaid
graph TD
    A[Input: 224x224x3] --> B[Conv2d: 3->32, 3x3, padding=1]
    B --> C[ReLU]
    C --> D[BatchNorm2d: 32]
    D --> E[MaxPool2d: 2x2]
    E --> F[Dropout: 0.25]
    F --> G[ResidualBlock: 32->64]
    G --> H[MaxPool2d: 2x2]
    H --> I[Dropout: 0.25]
    I --> J[ResidualBlock: 64->128]
    J --> K[MaxPool2d: 2x2]
    K --> L[Dropout: 0.25]
    L --> M[Conv2d: 128->128, 3x3, padding=1]
    M --> N[ReLU]
    N --> O[BatchNorm2d: 128]
    O --> P[MaxPool2d: 2x2]
    P --> Q[Dropout: 0.25]
    Q --> R[Flatten]
    R --> S[Linear: 128*14*14->512]
    S --> T[ReLU]
    T --> U[Dropout: 0.5]
    U --> V[Linear: 512->4]
    V --> W[Output: 4 classes]

    subgraph ResidualBlock 32->64
        G1[Conv2d: 32-->64, 3x3, padding=1] --> G2[BatchNorm2d]
        G2 --> G3[ReLU]
        G3 --> G4[Conv2d: 64->64, 3x3, padding=1]
        G4 --> G5[BatchNorm2d]
        G5 --> G6[Add: Shortcut]
        G7[Shortcut: Conv2d 1x1, BatchNorm] --> G6
    end

    subgraph ResidualBlock 64->128
        J1[Conv2d: 64->128, 3x3, padding=1] --> J2[BatchNorm2d]
        J2 --> J3[ReLU]
        J3 --> J4[Conv2d: 128->128, 3x3, padding=1]
        J4 --> J5[BatchNorm2d]
        J5 --> J6[Add: Shortcut]
        J7[Shortcut: Conv2d 1x1, BatchNorm] --> J6
    end
```

### Layer Details
- **Input**: 224x224 RGB images (3 channels)
- **Conv Block**:
  - Conv2d (3->32, 3x3, padding=1) → ReLU → BatchNorm → MaxPool (2x2) → Dropout (0.25)
  - ResidualBlock (32->64): 2x Conv2d (3x3) + BatchNorm + ReLU + Shortcut
  - MaxPool (2x2) → Dropout (0.25)
  - ResidualBlock (64->128): 2x Conv2d (3x3) + BatchNorm + ReLU + Shortcut
  - MaxPool (2x2) → Dropout (0.25)
  - Conv2d (128->128, 3x3, padding=1) → ReLU → BatchNorm → MaxPool (2x2) → Dropout (0.25)
- **Fully Connected**:
  - Flatten (128 * 14 * 14 = 25,088) → Linear (25,088->512) → ReLU → Dropout (0.5) → Linear (512->4)
- **Output**: Softmax probabilities for 4 classes

## Usage
1. **Prepare Dataset**: Ensure your MRI images are organized as described in [Dataset](#dataset).
2. **Train the Model**:
   ```bash
   python brain_tumor_detection.py
   ```
   - The script trains the model, saves the best weights (`best_model.pth`) based on validation loss, and evaluates on the test set.
   - Outputs include training/validation loss and accuracy, test metrics, classification report, and confusion matrix.
3. **Inference**:
   - Load the trained model for predictions:
     ```python
     model = CNNModel(num_classes=4).to(device)
     model.load_state_dict(torch.load("best_model.pth"))
     model.eval()
     # Add your inference code here
     ```

## Results
- **Test Accuracy**: 98.47%
- **Test Loss**: 0.0470
- **F1 Score**: 0.9847 (weighted)
- **Precision**: 0.9847 (weighted)
- **Recall**: 0.9847 (weighted)
- **Per-Class Metrics**:
  ```
  Class 0: Precision 0.99, Recall 0.99, F1 0.99
  Class 1: Precision 0.98, Recall 0.96, F1 0.97
  Class 2: Precision 0.98, Recall 1.00, F1 0.99
  Class 3: Precision 0.99, Recall 0.99, F1 0.99
  ```

### Training Curves
![Accuracy Plot](accuracy_plot.png)
![Loss Plot](loss_plot.png)

*Note*: Generate these plots by running the script, which saves them via matplotlib.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
