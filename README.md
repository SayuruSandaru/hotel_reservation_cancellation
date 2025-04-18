# Hotel Reservation Cancellation Prediction

This project predicts hotel booking cancellations using a deep learning model trained on a real-world dataset. It includes full data preprocessing, model creation, and evaluation — all built in Jupyter Notebooks.

## 🔍 Overview

Hotel cancellations can cause revenue loss and planning issues. This project uses customer, booking, and behavioral data to build a binary classifier that predicts whether a reservation will be canceled.

## 📁 Project Structure

- `Data_preprocessing_booking.ipynb` → cleans & encodes the raw dataset (`booking.csv`)
- `Model_creation_booking.ipynb` → builds, trains, and evaluates the deep learning model
- `cleaned_booking_data.csv` → preprocessed data used for training
- `booking.csv` → original dataset

## 📊 Dataset Features

- Booking details: number of nights, lead time, special requests
- Customer behavior: repeated bookings, previous cancellations
- Categorical fields: meal type, market segment, room type
- Target: `booking status_Not_Canceled` (1 = not canceled, 0 = canceled)

## 🧠 Model Overview

The model is built using **Keras**:
- Input: preprocessed features (scaled numeric, one-hot encoded categorical)
- Layers: Dense(64) → Dropout → Dense(32) → Dropout → Dense(1, sigmoid)
- Optimizer: Adam
- Loss: Binary crossentropy
- EarlyStopping used to prevent overfitting

## 📈 Results

- **Test Accuracy**: 85.6%
- **ROC AUC Score**: 0.92
- Evaluation includes:
  - Confusion Matrix
  - ROC Curve
  - Accuracy/Loss plots

