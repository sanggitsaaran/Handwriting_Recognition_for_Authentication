# 🔐 Handwriting Recognition for Authentication

**Secure signature-based authentication system using machine learning and computer vision**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-compatible-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

---

## 🎯 Overview

This project implements an intelligent **handwriting recognition and signature verification system** for biometric authentication. Leveraging advanced image processing and machine learning techniques, it classifies signatures as genuine or forged with high accuracy.

The system extracts sophisticated features from handwritten signatures and employs a **Softmax Regression model** trained on TensorFlow to make authentication decisions. It's ideal for secure document verification, banking applications, and fraud detection systems.

---

## ✨ Key Features

- **🖊️ Signature Verification**: Authenticate users based on handwritten signatures
- **🎨 Advanced Image Processing**: 
  - Noise denoising using FastNlMeansDenoising
  - Adaptive thresholding for robust binarization
  - Intelligent image cropping and normalization
- **🧠 Intelligent Feature Extraction**:
  - Flattened image features (32×10 dimension reduction)
  - Column-wise and row-wise density analysis
  - Aspect ratio computation for shape characteristics
- **📊 Comprehensive Evaluation**:
  - Confusion matrix generation
  - ROC-AUC curve analysis
  - Accuracy metrics and performance visualization
- **🔬 Machine Learning Ready**: Pre-configured with TensorFlow for quick training and deployment
- **📈 Scalable Architecture**: Easy to extend for multi-user authentication systems

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.7+ |
| **ML Framework** | TensorFlow 1.x |
| **Computer Vision** | OpenCV (cv2) |
| **Data Processing** | NumPy |
| **Metrics & Evaluation** | Scikit-learn, Matplotlib |
| **Model Type** | Softmax Regression |

---

## 🏗️ Project Architecture

```
Handwriting Recognition System
│
├── Image Input
│   ├── Preprocessing
│   │   ├── Denoising (FastNlMeansDenoising)
│   │   ├── Thresholding (Binary Inversion)
│   │   └── Cropping & Normalization
│   │
│   └── Feature Extraction
│       ├── Flattened Features (320 dims)
│       ├── Column Density (320 dims)
│       ├── Row Density (100 dims)
│       └── Aspect Ratio (1 dim)
│
├── Training Phase
│   ├── Load Training Data
│   ├── Extract Features
│   └── Train Softmax Model
│
├── Authentication/Testing
│   ├── Preprocess Test Image
│   ├── Extract Features
│   ├── Run Model Prediction
│   └── Generate Verdict (Genuine/Forged)
│
└── Evaluation
    ├── Confusion Matrix
    ├── ROC-AUC Curve
    └── Accuracy Report
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.7 or higher
- **pip**: Python package manager
- **Virtual Environment** (recommended): For isolated dependency management

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanggitsaaran/Handwriting_Recognition_for_Authentication.git
   cd Handwriting_Recognition_for_Authentication
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install opencv-python tensorflow numpy scikit-learn matplotlib
   ```

4. **Verify installation**
   ```bash
   python -c "import cv2, tensorflow, sklearn; print('All dependencies installed successfully!')"
   ```

---

## 🚀 Usage

### Running the Authentication System

```bash
python sign_verify.py
```

This command will:
1. Load training signatures from `data/training/[author_id]/`
2. Process and extract features from training data
3. Train the Softmax Regression model
4. Test against signatures in `data/test/[author_id]/`
5. Authenticate a test signature from `data/authenticating/[author_id]/`
6. Display confusion matrix and ROC-AUC curve
7. Print accuracy metrics

### Example Output

```
OpenCV version 4.5.2
authenticating....
authenticated
Image : test_genuine_1.png is  True [0 1]
Image : test_forged_1.png is  False [1 0]
...
Accuracy on test data: 94.32%
```

### Custom Configuration

To test with different signatures, modify these variables in `sign_verify.py`:

```python
author = '045'  # Change to different author ID
filename = 'test_2.png'  # Change test image filename
```

---

## 📁 Project Structure

```
Handwriting_Recognition_for_Authentication/
│
├── sign_verify.py                    # Main application (entry point)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
│
├── data/
│   ├── training/                     # Training signatures
│   │   └── [author_id]/
│   │       ├── genuine_1.png
│   │       ├── forged_1.png
│   │       └── ...
│   │
│   ├── test/                         # Test signatures for evaluation
│   │   └── [author_id]/
│   │       ├── genuine_test_1.png
│   │       ├── forged_test_1.png
│   │       └── ...
│   │
│   └── authenticating/               # Signatures for authentication
│       └── [author_id]/
│           ├── test_1.png
│           ├── test_2.png
│           └── ...
│
├── [author_genuine.zip]              # Sample signature datasets
└── [045-training.zip, 045-test.zip]  # Organized training/test sets
```

**How to Organize Your Data:**
- **Genuine signatures**: Name files with "genuine" in the filename
- **Forged signatures**: Use any other naming convention
- The system automatically labels based on filename patterns

---

## 🧠 How It Works

### 1️⃣ Image Preprocessing Pipeline

```
Raw Signature Image
    ↓
Convert to Grayscale (if needed)
    ↓
Apply Non-Local Means Denoising
    ↓
Binary Thresholding (Invert colors)
    ↓
Automatic Cropping to Content
    ↓
Processed Image Ready for Feature Extraction
```

**Preprocessing Steps:**
- **Denoising**: Removes noise while preserving signature details
- **Thresholding**: Creates binary image (foreground/background)
- **Cropping**: Removes excess whitespace, focuses on signature area
- **Normalization**: Standardizes image characteristics

### 2️⃣ Feature Extraction

For each preprocessed signature, the system extracts **741 features**:

| Feature Set | Dimensions | Description |
|-------------|-----------|-------------|
| Flattened Image | 320 | Resized signature (32×10) flattened |
| Column Density | 320 | Vertical stroke distribution (320×100 summed columns) |
| Row Density | 100 | Horizontal stroke distribution (320×100 summed rows) |
| Aspect Ratio | 1 | Width-to-height ratio of signature |
| **Total** | **741** | Complete feature vector |

### 3️⃣ Model Training & Classification

**Algorithm**: Softmax Regression (Multinomial Logistic Regression)

```
Training Loss: Cross-Entropy
Optimizer: Gradient Descent (learning rate: 0.1)
Iterations: Single epoch (optimizable)

y = softmax(W·x + b)
where:
  x = feature vector (741 dims)
  W = weight matrix (741×2)
  b = bias vector (2 dims)
  y = [P(forged), P(genuine)]
```

**Prediction Rule:**
- If `argmax(y) == 1` → **Genuine Signature** ✓
- If `argmax(y) == 0` → **Forged Signature** ✗

---

## 📈 Model Details

### Architecture

```
Input Layer (741 features)
    ↓
Softmax Regression
    ├─ Weights: 741×2 matrix
    ├─ Bias: 2-dimensional vector
    └─ Activation: Softmax
        ↓
Output Layer (2 classes)
    ├─ Class 0: Forged Signature
    └─ Class 1: Genuine Signature
```

### Training Configuration

- **Model Type**: Softmax Regression
- **Loss Function**: Cross-Entropy
- **Optimizer**: Gradient Descent
- **Learning Rate**: 0.1
- **Features**: 741-dimensional vectors
- **Output Classes**: 2 (Genuine/Forged)

### Why Softmax Regression?

✓ Interpretable probability outputs  
✓ Efficient for binary classification  
✓ Fast training on moderate datasets  
✓ Works well with engineered features  
✓ Suitable for real-time authentication  

---


## 📋 Data Format

### Input Image Requirements

| Property | Specification |
|----------|---------------|
| Format | PNG, JPG/JPEG, BMP |
| Dimensions | Flexible (auto-normalized) |
| Mode | RGB or Grayscale |
| DPI | No specific requirement |
| Size | Recommended: 200-500px width |

### File Naming Convention

```
# Genuine signatures (must contain "genuine")
genuine_user_1.png
genuine_signature_001.png
my_genuine_sig.png

# Forged signatures (anything without "genuine")
forged_user_1.png
fake_001.png
attempt_2.png
```

### Dataset Organization

```
data/
├── training/045/
│   ├── genuine_1.png
│   ├── genuine_2.png
│   ├── forged_1.png
│   └── forged_2.png
│
├── test/045/
│   ├── genuine_test_1.png
│   ├── forged_test_1.png
│   └── ...
│
└── authenticating/045/
    ├── test_1.png
    └── test_2.png
```

---

## 🔬 Extended Usage Examples

### Example 1: Single Signature Verification

```python
from sign_verify import imageprep, runmodel
import cv2

# Load and preprocess a single signature
sig_image = cv2.imread('signature.png', 0)
features = imageprep(sig_image)

# Verify against training data
is_genuine = verify_signature(features, training_data, training_labels)
print("Verified!" if is_genuine else "Not verified")
```

### Example 2: Batch Processing

```bash
# Process multiple signatures in the test directory
python sign_verify.py
# Generates evaluation metrics for all test images
```

### Example 3: Custom Author Training

Edit `sign_verify.py`:
```python
author = 'your_user_id'  # Change this
# System will automatically load from data/training/your_user_id/
```

---

## 🎓 Educational Value

This project demonstrates:
- **Computer Vision**: Image preprocessing and feature extraction
- **Machine Learning**: Supervised classification with TensorFlow
- **Biometric Authentication**: Real-world security applications
- **Performance Evaluation**: Metrics and visualization
- **Software Engineering**: Project structure and modularity

Perfect for understanding:
- Neural network basics
- Image processing pipelines
- Authentication systems
- Fraud detection mechanisms

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

**You are free to:**
- ✅ Use commercially
- ✅ Modify the code
- ✅ Distribute copies
- ✅ Private use

---

## 👤 Contact & Support

**Project Author**: Sanggit Saaran  
**Repository**: [GitHub - Handwriting_Recognition_for_Authentication](https://github.com/sanggitsaaran/Handwriting_Recognition_for_Authentication)

### Connect With Us

- 💼 LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/sanggit-saaran-k-c-s/)
- 📧 Email: cb.sc.u4aie23247@cb.students.amrita.edu

### Questions & Issues

- Have a question? Open a [GitHub Discussion](https://github.com/sanggitsaaran/Handwriting_Recognition_for_Authentication/discussions)
- Found a bug? Submit a [GitHub Issue](https://github.com/sanggitsaaran/Handwriting_Recognition_for_Authentication/issues)

---

## 🙏 Acknowledgments

- **OpenCV Community** for computer vision capabilities
- **TensorFlow Team** for ML frameworks
- **Scikit-learn** for metrics and evaluation tools

---

## 📚 References & Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Guides](https://www.tensorflow.org/guide)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Signature Verification Research](https://en.wikipedia.org/wiki/Signature_recognition)

---

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star! ⭐</b>
</p>

<p align="center">
  Made with ❤️ for secure authentication
</p>

---

**Last Updated**: March 2026  