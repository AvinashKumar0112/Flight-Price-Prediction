
## Flight Price Prediction using AWS SageMaker

## Introduction
This project aims to predict flight prices based on various features such as departure time, arrival time, airline, and more. Utilizing AWS SageMaker, we develop, train, and deploy a machine learning model that helps in forecasting flight prices with high accuracy. This project demonstrates the end-to-end process of building a machine learning solution on the cloud.


## Project Overview
The project consists of the following steps:

**1. Data Collection and Preprocessing:**
- Collected data on flight prices and relevant features.
- Cleaned and preprocessed the data using Pandas and NumPy.

**2. Exploratory Data Analysis (EDA):**
- Conducted a thorough EDA to understand the distribution of data, identify patterns, and spot anomalies.
- Visualized key relationships between features and the target variable.

**3. Feature Engineering:**
- Engineered new features to improve model performance.
- Applied techniques like encoding categorical variables and scaling numerical features using Scikit-learn.

**4. Model Training:**
- Trained multiple machine learning models using AWS SageMaker, including Linear Regression, Decision Trees, and XGBoost.
- Tuned hyperparameters to optimize model performance.

**5. Model Deployment:**
- Deployed the best-performing model on AWS SageMaker.
- Created a simple web application using Streamlit to demonstrate the model’s predictions.
## Technology used: 
- **AWS SageMaker**
- **Python:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **IDE:** VS Code, Jupyter Notebook 
- **Streamlit:** For web application development
- **AWS S3:** For data storage
- **AWS EC2:** For computing resources


## Installation and Setup
1. Clone the repository:

```
https://github.com/AvinashKumar0112/Flight-Price-Prediction.git
```
2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Set up your AWS environment:

- Configure AWS CLI with your credentials.
- Ensure you have access to AWS SageMaker, S3, and EC2.
- Run the web application:

```
streamlit run app.py
```

- OR you can directly run the web application through below link:
```
https://sagemaker-flight-prices-prediction-5yfn7cnnpxta2k23jvf4ic.streamlit.app/
```


## Usage
- Load the data and run the preprocessing scripts.
- Execute the model training notebooks to train and evaluate different models.
- Use the deployment script to deploy the model to AWS SageMaker.
- Access the Streamlit web app to interact with the model and make predictions.


## Results
The model achieved an RMSE (Root Mean Square Error) of X on the test set, indicating a high level of accuracy in predicting flight prices. Below are some visualizations of the model’s performance:


## Conclusion
This project demonstrates how to leverage AWS SageMaker for building and deploying machine learning models in a scalable and efficient manner. Future improvements could include incorporating additional features or experimenting with more complex models.
