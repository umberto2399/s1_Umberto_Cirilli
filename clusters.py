import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('telco.csv')

# Drop columns that can cause data leakage
leakage_columns = ['Churn Score', 'Customer Status', 'Churn Category', 'Churn Reason', 'Churn Label']
data.drop(columns=leakage_columns, inplace=True)

# Identify numerical features for clustering
numerical_features = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges', 'Latitude', 'Longitude']

# Ensure numerical features have no missing values
data = data.dropna(subset=numerical_features)

# Perform KMeans clustering on the numerical features
kmeans = KMeans(n_clusters=3, random_state=0)
data['Cluster'] = kmeans.fit_predict(data[numerical_features])

# Save the updated dataset to a new CSV file
data.to_csv('telco_with_clusters.csv', index=False)

print("Clustering completed and data saved to 'telco_with_clusters.csv'.")