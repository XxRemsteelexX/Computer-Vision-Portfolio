# Computer Vision Portfolio

Advanced computer vision portfolio featuring neural network architectures and deep learning implementations for image classification and landmark recognition.

## Projects

### 1. Landmark Classification
**Location**: `projects/landmark-classification/`

A comprehensive deep learning project for landmark recognition using convolutional neural networks (CNNs).

**Features:**
- Custom CNN architecture with 4 convolutional blocks
- Batch normalization and dropout for regularization
- Transfer learning implementation
- Data preprocessing and augmentation
- Model training and evaluation pipeline

**Technology Stack:**
- PyTorch for deep learning framework
- Custom CNN architecture design
- Data loaders and transformations
- Model checkpointing and evaluation

**Key Components:**
- `src/model.py` - CNN architecture definition
- `src/transfer.py` - Transfer learning implementation
- `src/data.py` - Data loading and preprocessing
- `src/helpers.py` - Utility functions
- `src/create_submit_pkg.py` - Model packaging

## Technical Skills Demonstrated

- **Deep Learning Frameworks**: PyTorch
- **Neural Network Architectures**: CNNs, Transfer Learning
- **Computer Vision**: Image Classification, Feature Extraction
- **Model Optimization**: Batch Normalization, Dropout, Adaptive Pooling
- **Data Processing**: Image Preprocessing, Data Augmentation
- **Model Deployment**: Model Packaging and Submission

## Architecture Highlights

### Custom CNN Model
- **Input**: 3-channel RGB images
- **Architecture**: 4 convolutional blocks with increasing depth (64→128→256→512)
- **Regularization**: Batch normalization and dropout
- **Pooling**: MaxPooling and Adaptive Average Pooling
- **Output**: Configurable number of classes (default 1000)

### Key Features
- Progressive feature extraction with deeper layers
- Batch normalization for training stability
- Configurable dropout for overfitting prevention
- Adaptive pooling for flexible input sizes

## Project Structure
```
projects/
└── landmark-classification/
    └── src/
        ├── model.py          # CNN architecture
        ├── transfer.py       # Transfer learning
        ├── data.py           # Data processing
        ├── helpers.py        # Utilities
        └── create_submit_pkg.py # Model packaging
```

## Future Enhancements

- GAN implementations for image generation
- Object detection with YOLO/R-CNN
- Semantic segmentation projects
- Real-time computer vision applications
- Model optimization and quantization

---

*This portfolio demonstrates production-ready computer vision implementations with modern deep learning techniques.*