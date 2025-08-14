# Hotel Reservation Predictor App
Machine learning model and interactive **Streamlit application** to predict hotel booking cancellations based on guest, booking, and pricing details. The project covers data preprocessing, model training, evaluation, saving model artifacts, and building a user-friendly web app for real-time predictions.

---

## Overview
This project was developed for the **Model Deployment** course (4th semester) and focuses on building a predictive model to determine whether a hotel booking will be canceled or not.  

The workflow includes:
- Data cleaning and preprocessing (handling missing values, encoding categorical variables, scaling, and balancing classes)
- Model experimentation and evaluation (XGBoost vs Random Forest)
- Saving the final model and preprocessing objects for deployment
- Preparing test cases for a Streamlit application

**Deliverables include:**
- **Python notebook** – complete preprocessing, model training, and evaluation pipeline
- **Saved model artifacts** – `.pkl` files for deployment
- **Streamlit-ready input format** – for real-time prediction
- **Documentation** – step-by-step project explanation

---

## 📊 Data

The dataset contains **36,275 entries** and **18 features** describing hotel booking details, customer information, and pricing data.

**Key Information**
- **Entries:** 36,275 rows  
- **Features:** 18 columns (13 numerical, 3 categorical, 1 ID column, 1 target variable)  
- **Target Variable:** `booking_status` (Canceled / Not_Canceled)  

### Data Dictionary

| Variable | Description |
|----------|-------------|
| **Booking_ID** | Unique identifier for each booking |
| **no_of_adults** | Number of adults in the booking |
| **no_of_children** | Number of children in the booking |
| **no_of_weekend_nights** | Number of weekend nights booked |
| **no_of_week_nights** | Number of week nights booked |
| **type_of_meal_plan** | Meal plan chosen by the guest |
| **required_car_parking_space** | Whether a parking space was requested |
| **room_type_reserved** | Type of room reserved |
| **lead_time** | Number of days between booking and check-in |
| **arrival_year** | Year of arrival |
| **arrival_month** | Month of arrival |
| **arrival_date** | Date of arrival |
| **market_segment_type** | Market segment classification |
| **avg_price_per_room** | Average room price per night |
| **no_of_special_requests** | Number of special requests made |
| **booking_status** | Booking outcome: Canceled or Not_Canceled |

---

## 🚀 Features
- Data preprocessing pipeline:
  - Missing value removal
  - One-hot encoding for categorical features
  - Robust scaling for numerical features
  - Class balancing with RandomOverSampler
- Model comparison:
  - **XGBoost Classifier** vs **Random Forest Classifier**
- Performance metrics:
  - Accuracy, Precision, Recall, F1-score, AUC
- Model persistence:
  - Saving model and preprocessing objects in `.pkl` format
- Deployment-ready:
  - Test cases prepared for Streamlit app validation

---

## 🛠️ Tools & Technologies
- **Python** – data preprocessing and model training
- **Pandas / NumPy** – data manipulation
- **Matplotlib / Seaborn** – data visualization
- **Scikit-learn** – preprocessing, evaluation, and metrics
- **XGBoost** – gradient boosting model
- **Random Forest** – ensemble model for final deployment
- **Imbalanced-learn** – RandomOverSampler for class balancing
- **Pickle & gzip** – model and object serialization
- **Streamlit** – interactive deployment interface
