Overview

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify plant leaf diseases using the Leaf Disease Dataset (Combination). The model supports early detection of diseases such as:

Early Blight

Late Blight

And more (38 classes total)

This helps farmers manage crop health effectively.

Features

CNN Architecture: 2 convolutional layers + adaptive pooling + fully connected layers

Image preprocessing: Resize to 128x128, normalization

Data augmentation: Random horizontal flip & rotation (±10°)

Adaptive pooling: Handles variable image sizes

Evaluation tools: Loss curves, confusion matrix, classification report

Dataset

Leaf Disease Dataset (Combination)

Number of classes: 38

Folder structure:

dataset/
    train/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/
        ...

Hyperparameters

Batch size: 32

Number of epochs: 10

Learning rate: 0.001

Installation & Usage

Clone the repository

git clone https://github.com/your-username/plant-leaf-disease-classification.git
cd plant-leaf-disease-classification


Create and activate a virtual environment

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies

pip install -r requirements.txt


Example requirements.txt:

torch
torchvision
matplotlib
seaborn
scikit-learn
numpy


Update dataset paths in train.py

train_dataset = torchvision.datasets.ImageFolder(root="path_to_train", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root="path_to_test", transform=transform)


Run training & evaluation

python train.py


Outputs

Training vs Validation Loss Curve

Confusion Matrix

Classification Report (precision, recall, F1-score per class)

CNN Model Architecture
Input: 128x128 RGB Images
→ Conv2d(3,32,3) + ReLU → MaxPool2d(2)
→ Conv2d(32,64,3) + ReLU → MaxPool2d(2)
→ AdaptiveAvgPool2d((7,7))
→ Flatten
→ Linear(64*7*7 → 256) + ReLU
→ Linear(256 → 38)
Output: Class Probabilities

Results

High accuracy in detecting plant leaf diseases

Confusion matrix & classification report visualize model performance



Training vs Validation Loss Curve:


Future Improvements

Implement advanced architectures like ResNet or EfficientNet

Real-time detection via mobile apps or drones

Expand dataset with more plant types & disease categories

License

MIT License
