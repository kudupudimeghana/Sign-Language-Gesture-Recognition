# Sign Language Gesture Recognition System

A real-time sign language gesture recognition application built using Python, OpenCV, MediaPipe and Machine Learning. The system detects hand gestures through a webcam, extracts hand landmarks and predicts corresponding signs.

## Problem Statement

Communication can be challenging between people who use sign language and those who are unfamiliar with it. This project explores how computer vision and machine learning can help recognize hand gestures and convert them into readable text.

## Project Objectives

- Detect and track hand movements in real time.
- Extract hand landmarks for gesture analysis.
- Train a machine learning model to classify hand gestures.
- Display recognized gestures as text.

## Features

- Real-time webcam-based hand detection.
- Extraction of 21 hand landmarks using MediaPipe.
- Machine learning-based gesture classification.
- Recognition of five predefined gestures:
  - HELLO
  - YES
  - NO
  - OK
  - STOP

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| OpenCV | Webcam input and image processing |
| MediaPipe | Hand landmark detection |
| Scikit-learn | Machine learning classification |
| NumPy | Numerical operations |
| Pandas | Dataset handling |

## System Workflow

1. Capture live video using a webcam.
2. Detect the hand using MediaPipe.
3. Extract 21 hand landmarks.
4. Process the extracted landmark coordinates.
5. Pass the features to the trained Random Forest model.
6. Predict the gesture and display its corresponding text.

## Project Structure

```text
Sign-Language-Gesture-Recognition/
│
├── dataset/
├── models/
├── screenshots/
├── src/
│   ├── collect_data.py
│   ├── hand_detection.py
│   ├── predict_gesture.py
│   ├── train_model.py
│   └── webcam_test.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/kudupudimeghana/Sign-Language-Gesture-Recognition.git
```

**2. Navigate to the project directory**

```bash
cd Sign-Language-Gesture-Recognition
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the environment**

Windows:

```bash
venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the application**

Use the appropriate Python script from the `src/` directory to collect data, train the model or test gesture recognition.

## Future Enhancements

- Integrate Text-to-Speech to convert recognized gestures into spoken audio.
- Expand the gesture vocabulary.
- Explore deep learning and sequence-based gesture recognition.
- Develop a user-friendly application interface.
- Explore continuous sign language translation.

## Project Motivation

This project is an exploration of how artificial intelligence and computer vision can contribute to accessible communication. It also provides an opportunity to learn practical machine learning, data processing and real-time
