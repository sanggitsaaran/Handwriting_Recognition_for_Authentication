# Handwriting Recognition for Authentication

## Project Overview

**Handwriting Recognition for Authentication** is an innovative project that integrates computer vision and machine learning to build a secure signature-based authentication system. Developed using TensorFlow and OpenCV, this project addresses the growing demand for fraud detection and reliable identity verification. By automating the classification of handwritten signatures as genuine or forged, the system enhances security protocols and reduces the burden of manual verification.

## Team Members

* Sanggit Saaran K C S (CB.SC.U4AIE23247)
* Surya Ha (CB.SC.U4AIE23267)
* Vishal Seshadri B (CB.SC.U4AIE23260)
* Venkatram K S (CB.SC.U4AIE23236)
* Harivaarthan T D (CB.SC.U4AIE23228)

## Abstract

This project focuses on the development of a signature authentication system using handwriting recognition. By leveraging machine learning models and image processing, the system differentiates between genuine and forged signatures. The workflow includes dataset creation, image preprocessing, feature extraction, model training, and performance evaluation. The goal is to provide a scalable and intelligent solution to combat signature forgery in real-world applications.

## Introduction

Signature verification plays a crucial role in various sectors such as banking, legal documentation, and secure authentication systems. Manual verification is often time-consuming and prone to error. This project seeks to automate the process using supervised machine learning techniques to enhance both accuracy and efficiency. By analyzing the intrinsic patterns in handwriting, the model learns to identify subtle differences that distinguish genuine signatures from forgeries.

## Dataset and Preprocessing

* **Dataset Creation**: A diverse set of signature samples was collected to reflect individual variation and potential forgery attempts.
* **Labeling**: Each signature was labeled as either `genuine` or `forged`, enabling supervised learning.
* **Image Processing**: OpenCV was used for tasks such as grayscale conversion, thresholding, resizing, and noise reduction to prepare the data for model input.
* **Feature Extraction**: Relevant features such as stroke dynamics, contours, and geometric shapes were extracted to aid classification.

## Model Architecture and Training

* **Model Used**: TensorFlow's SoftMax regression model.
* **Training Objective**: To classify signatures based on learned patterns from labeled data.
* **Learning Method**: Supervised learning with cross-entropy loss minimization.
* **Training Pipeline**: Data normalization → Feature vector generation → Model fitting.

## Evaluation Metrics

* **Confusion Matrix**: Used to measure the true positive, false positive, true negative, and false negative rates.
* **ROC Curve**: Visualized model performance in distinguishing between classes across various threshold values.
* **Accuracy Score**: Evaluated model's overall prediction correctness.

## Results and Analysis

The system showed promising results in correctly classifying signature images. The confusion matrix indicated high precision and recall for the genuine class, while the ROC curve revealed strong model sensitivity and specificity. Despite occasional false positives/negatives, the model demonstrated its capability to be integrated into real-world authentication systems with minimal supervision.

## Conclusions

This project demonstrates the potential of combining computer vision with machine learning to automate and improve signature verification. While the model effectively reduces manual effort and increases reliability, it also highlights the need for continuous retraining to adapt to evolving handwriting patterns. As AI-based systems become more widespread, such innovations contribute significantly to secure, scalable, and intelligent authentication frameworks.
