# Telco Customer Churn Prediction

This repository contains a Streamlit application for predicting customer churn in a telco dataset. The app includes functionalities for filtering and displaying data, providing insights and visualizations, predicting churn for new customers, and making city-based predictions with clustering analysis.

[LINK TO SEE IMPROVEMENTS](https://drive.google.com/drive/folders/1aXjUMYx8VTlIlIYtPOmRzY9vSETQMPgq?usp=drive_link)
 

## Repository Structure

- **app.py**: The main Streamlit application script.
- **model.py**: Script for training the churn prediction model using a RandomForest classifier.
- **clusters.py**: Script for performing KMeans clustering on the dataset.
- **churn_model_pipeline.pkl**: The trained pipeline model for predicting customer churn.
- **requirements.txt**: List of required Python packages for the project.
- **telco.csv**: Original dataset containing telco customer data.
- **telco_with_clusters.csv**: Dataset with added cluster labels from KMeans clustering.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/telco-churn-prediction.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd telco-churn-prediction
    ```

3. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

4. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Set up your OpenAI API key as an environment variable:**
    ```bash
    export OPENAI_API_KEY='your_api_key_here'
    ```

## Usage

### 1. Train the Model
To train the model, run the `model.py` script. This will train a RandomForest classifier on the telco dataset and save the trained model as `churn_model_pipeline.pkl`.

```bash
python model.py
```

### 2. Perform Clustering
To perform clustering on the dataset, run the `clusters.py` script. This will add cluster labels to the dataset and save the updated dataset as `telco_with_clusters.csv`.

```bash
python clusters.py
```

### 3. Run the Streamlit App
To run the Streamlit application, execute the `app.py` script.

```bash
streamlit run app.py
```

## Application Features

### Filter and Display Data
- Filter data by predicted churn, churn probability, city, contract type, internet type, and payment method.
- Display filtered data in a table format and download it as a CSV file.
- Visualize customers on a map based on their churn status.

### Insights and Visualizations
- Display distributions of numerical features.
- Show relationships between features and churn.
- Provide a correlation matrix of numerical features.

### Predict Churn for a New Customer
- Input new customer details to predict churn.
- Display the predicted churn label and probability.
- Generate recommendations using OpenAI's GPT-4 based on the customer characteristics and predicted churn.

### City-Based Predictions
- Select a city to predict churn for all customers in that city.
- Display churn statistics for the selected city.
- Show detailed customer data and predictions.
- Visualize clusters on a map.
- Generate cluster-wise statistics and recommendations using OpenAI's GPT-4, considering the percentage of users in each cluster.

## Adjustments Made Based on Feedback
- **User-Centric Improvements**: Added functionality to focus on specific cities and high churn probability customers.
- **Enhanced Map Visualization**: Improved map zoom levels to provide a better user experience.
- **Cluster Analysis**: Incorporated KMeans clustering to segment customers and provide tailored recommendations.
- **Recommendations Integration**: Used OpenAI's GPT-4o to generate actionable recommendations based on customer data and clustering insights.
- **Streamlined Predictions**: Added city-based prediction features to help managers understand and act on churn risks at a city level.

## Dataset Description
The dataset (`telco.csv`) contains customer data from a telecommunications company, including demographic information, service details, and churn status. The processed dataset (`telco_with_clusters.csv`) includes additional cluster labels generated through KMeans clustering.

## Requirements
Ensure all the necessary Python packages listed in `requirements.txt` are installed.

---

