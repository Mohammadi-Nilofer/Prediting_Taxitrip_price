# Taxi Trip Price Prediction 🚕  

##  Project Overview  
This project focuses on predicting **taxi trip fares** using **Artificial Neural Networks (ANNs)**. Taxi pricing is influenced by multiple factors such as trip distance, time of day, traffic, and weather conditions. By building a robust regression model, this project aims to enhance pricing accuracy, benefiting both customers and taxi operators.  

##  Problem Statement  
Accurately predicting taxi fares remains challenging due to the dynamic nature of traffic, weather, and other conditions. Traditional fare structures often lead to inconsistencies and inefficiencies. This project leverages **machine learning** to build a model capable of delivering more reliable fare estimates.  

##  Objectives  
- Build and optimize an **ANN Regressor** for taxi fare prediction.  
- Perform **data preprocessing**, **feature engineering**, and **exploratory data analysis (EDA)**.  
- Conduct **hyperparameter tuning** for model optimization.  
- Evaluate the model using regression metrics such as **RMSE** and **R²-score**.  

## 📂 Dataset  
- **Source**: [Kaggle – Taxi Price Prediction](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction)  
- **Features**:  
  - Distance (km)  
  - Pickup Time & Dropoff Time  
  - Traffic Condition (light, medium, heavy)  
  - Passenger Count  
  - Weather Condition (clear, rain, snow)  
  - Trip Duration (minutes)  
- **Target**: Fare Amount (USD)  

## 🛠️ Methodology  
1. **Data Preprocessing**  
   - Handling missing values, encoding categorical variables, and scaling numerical features.  
   - Feature engineering (e.g., extracting time-based features).  

2. **Exploratory Data Analysis (EDA)**  
   - Visualizing fare distribution, correlations, and impact of traffic/weather.  

3. **Modeling**  
   - Built an **Artificial Neural Network (ANN) Regressor** using deep learning frameworks.  
   - Performed **hyperparameter tuning** to optimize architecture, activation functions, learning rate, and epochs.  

4. **Evaluation**  
   - Metrics: **Root Mean Squared Error (RMSE)** and **R²-score**.  
   - Compared different configurations to select the best model.  

## 📊 Results  
- The optimized ANN model achieved strong performance in predicting taxi fares.  
- Demonstrated the importance of **traffic** and **weather conditions** in influencing fare predictions.  
- Model evaluation confirmed its ability to generalize well to unseen data.  

##  Tech Stack  
- **Programming**: Python  
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras  
- **Tools**: Jupyter Notebook  

##  Future Improvements  
- Integrating **real-world GPS and traffic API data**.  
- Deploying the model via **Streamlit/Flask** for real-time predictions.  
- Experimenting with advanced architectures like **LSTM (time-series)** for dynamic pricing.  

## 🙌 Acknowledgements  
Special thanks to the dataset contributors on Kaggle and the open-source community for supporting tools and libraries.  
