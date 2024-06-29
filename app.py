import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import pydeck as pdk
import openai as OpenAI
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained pipeline model
pipeline = joblib.load('churn_model_pipeline.pkl')

# Load the dataset with clusters
data = pd.read_csv('telco_with_clusters.csv')

# Ensure features are defined
leakage_columns = ['Churn Score', 'Customer Status', 'Churn Category', 'Churn Reason', 'Churn Label']
features = [col for col in data.columns if col not in leakage_columns + ['Customer ID', 'Cluster']]
numerical_features = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges', 'Latitude', 'Longitude']
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Apply the model to the entire DataFrame to predict churn
predict_features = [col for col in features if col != 'Cluster']
data['Predicted Churn'] = pipeline.predict(data[predict_features])
data['Churn Probability'] = pipeline.predict_proba(data[predict_features])[:, 1]
data['Predicted Churn'] = data['Predicted Churn'].map({1: 'Yes', 0: 'No'})

# OpenAI API setup
client = OpenAI.Client(api_key='Your_API_Key_Here')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Filter and Display Data", "Insights and Visualizations", "Predict Churn for a New Customer", "City-Based Predictions"])

# Section 1: Filter and Display Data
if page == "Filter and Display Data":
    st.title("Filter and Display Data")
    
    # Filter by predicted churn
    predicted_churn_filter = st.selectbox("Filter by Predicted Churn", ["All", "Yes", "No"])
    
    # Filter by churn probability
    churn_probability_range = st.slider("Churn Probability Range", 0.0, 1.0, (0.0, 1.0))
    
    # Select number of rows to display
    rows_to_display = st.selectbox("Select number of rows to display", ["All rows", 25, 50, 100])
    
    # Select city filter
    cities = ["All cities"] + data['City'].unique().tolist()
    selected_city = st.selectbox("Select City", cities)
    
    # Select contract type filter
    contract_types = ["All contract types"] + data['Contract'].unique().tolist()
    selected_contract = st.selectbox("Select Contract Type", contract_types)
    
    # Select internet type filter
    internet_types = ["All internet types"] + data['Internet Type'].unique().tolist()
    selected_internet_type = st.selectbox("Select Internet Type", internet_types)
    
    # Select payment method filter
    payment_methods = ["All payment methods"] + data['Payment Method'].unique().tolist()
    selected_payment_method = st.selectbox("Select Payment Method", payment_methods)
    
    # Filter data based on widget selections
    filtered_data = data.copy()
    if predicted_churn_filter != "All":
        filtered_data = filtered_data[filtered_data['Predicted Churn'] == predicted_churn_filter]
    filtered_data = filtered_data[(filtered_data['Churn Probability'] >= churn_probability_range[0]) & (filtered_data['Churn Probability'] <= churn_probability_range[1])]
    if selected_city != "All cities":
        filtered_data = filtered_data[filtered_data['City'] == selected_city]
    if selected_contract != "All contract types":
        filtered_data = filtered_data[filtered_data['Contract'] == selected_contract]
    if selected_internet_type != "All internet types":
        filtered_data = filtered_data[filtered_data['Internet Type'] == selected_internet_type]
    if selected_payment_method != "All payment methods":
        filtered_data = filtered_data[filtered_data['Payment Method'] == selected_payment_method]
    
    # Display the filtered DataFrame
    if rows_to_display != "All rows":
        st.dataframe(filtered_data[['Customer ID'] + features + ['Predicted Churn', 'Churn Probability']].head(rows_to_display))
    else:
        st.dataframe(filtered_data[['Customer ID'] + features + ['Predicted Churn', 'Churn Probability']])
    
    # Download button for filtered data
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv',
    )
    
    # Plot customers on a map with colors based on predicted churn label
    map_data = filtered_data[['Latitude', 'Longitude', 'Predicted Churn']]
    map_data['color'] = map_data['Predicted Churn'].apply(lambda x: [255, 0, 0] if x == 'Yes' else [0, 0, 255])
    
    # Automatically adjust zoom level based on the density of points
    if not map_data.empty:
        lat_mean = map_data['Latitude'].mean()
        lon_mean = map_data['Longitude'].mean()
        zoom_level = 10 if len(map_data) < 100 else 5 if len(map_data) < 1000 else 3
        
        # Create a pydeck layer for the map
        layer = pdk.Layer(
            'ScatterplotLayer',
            map_data,
            get_position='[Longitude, Latitude]',
            get_fill_color='color',
            get_radius=200,
            pickable=True,
            auto_highlight=True
        )
        
        # Determine the initial view state for the map
        initial_view_state = pdk.ViewState(
            latitude=lat_mean,
            longitude=lon_mean,
            zoom=zoom_level,
            pitch=50
        )
        
        # Create the deck.gl map with a lighter style
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=initial_view_state,
            tooltip={"text": "Churn: {Predicted Churn}\nLatitude: {Latitude}\nLongitude: {Longitude}"},
            map_style='mapbox://styles/mapbox/light-v10'
        )
        
        # Display the map
        st.pydeck_chart(r)

    # Search by Customer ID section
    st.subheader("Search by Customer ID")
    customer_id = st.text_input("Enter Customer ID to search", value='')

    # Display customer information and prediction if a valid ID is entered
    if customer_id:
        customer_data = data[data['Customer ID'] == customer_id]
        if not customer_data.empty:
            st.write("Customer Information")
            st.write(customer_data[['Customer ID', 'Age', 'City', 'State', 'Monthly Charge', 'Total Charges', 'Predicted Churn', 'Churn Probability']])

            # Display churn prediction
            churn_prediction = customer_data['Predicted Churn'].values[0]
            churn_probability = customer_data['Churn Probability'].values[0]
            st.write('Churn Prediction:', churn_prediction)
            st.write('Churn Probability:', churn_probability)
        else:
            st.write("Customer ID not found.")

# Section 2: Insights and Visualizations
elif page == "Insights and Visualizations":
    st.title("Insights and Visualizations")

    # Widget: Distribution of numerical features
    st.subheader("Distribution of Numerical Features")
    num_feature = st.selectbox("Select a numerical feature", numerical_features)
    fig, ax = plt.subplots()
    sns.histplot(data[num_feature], kde=True, ax=ax)
    st.pyplot(fig)

    # Widget: Relationship between features and churn
    st.subheader("Relationship Between Features and Churn")
    feature = st.selectbox("Select a feature", features)
    if feature in numerical_features:
        fig, ax = plt.subplots()
        sns.boxplot(x=data['Predicted Churn'], y=data[feature], ax=ax)
        st.pyplot(fig)
    elif feature in categorical_features:
        fig, ax = plt.subplots()
        churn_counts = data.groupby([feature, 'Predicted Churn']).size().unstack().fillna(0)
        churn_counts.plot(kind='bar', stacked=True, ax=ax)
        st.pyplot(fig)

    # Correlation matrix section
    st.subheader("Correlation Matrix")
    corr_matrix = data[numerical_features + ['Churn Probability']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Section 3: Predict Churn for a New Customer
elif page == "Predict Churn for a New Customer":
    st.title("Predict Churn for a New Customer")

    # Create input fields for all predictors except 'Customer ID'
    input_data = {}
    for feature in features:
        if feature in categorical_features:
            unique_values = data[feature].unique()
            if len(unique_values) == 2:  # Binary categorical feature
                input_data[feature] = st.radio(f"Select {feature}", options=unique_values)
            else:  # Non-binary categorical feature
                input_data[feature] = st.selectbox(f"Select {feature}", options=unique_values)
        elif feature in ['Zip Code', 'Latitude', 'Longitude']:
            input_data[feature] = st.number_input(f"Enter {feature}", value=float(data[feature].mean()))
        else:  # All other numeric features
            min_value = int(data[feature].min())
            max_value = int(data[feature].max())
            mean_value = int(data[feature].mean())
            input_data[feature] = st.slider(f"Select {feature}", min_value, max_value, mean_value)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict button for new customer churn prediction
    if st.button("Predict Churn"):
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[:, 1][0]
        churn_label = 'Yes' if prediction[0] == 1 else 'No'
        st.write('Churn Prediction:', churn_label)
        st.write('Churn Probability:', probability)

        # Formulate the prompt for GPT-4
        prompt = (
            f"Customer characteristics:\n{input_data}\n\n"
            f"Predicted Churn Label: {churn_label}\n"
            f"Churn Probability: {probability}\n\n"
            "Based on these details, what should a telco manager do to address this customer's needs and potentially reduce the risk of churn?"
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a customer service expert providing recommendations to reduce churn based on customer data."},
                {"role": "user", "content": prompt}
            ]
        )

        recommendations = response.choices[0].message.content.strip()
        st.write('Recommendations:', recommendations)

# Section 4: City-Based Predictions
elif page == "City-Based Predictions":
    st.title("City-Based Predictions")

    # Widget: Select city filter for predictions
    cities = data['City'].unique().tolist()
    selected_city = st.selectbox("Select City for Predictions", cities)

    if st.button("Predict Churn for Selected City"):
        city_data = data[data['City'] == selected_city]
        if not city_data.empty:
            churn_counts = city_data['Predicted Churn'].value_counts(normalize=True) * 100
            st.write(f"Churn Prediction for {selected_city}")
            st.write(f"Percentage of customers predicted to churn: {churn_counts.get('Yes', 0):.2f}%")
            st.write(f"Percentage of customers predicted not to churn: {churn_counts.get('No', 0):.2f}%")

            st.write("Customer Data and Predictions")
            st.dataframe(city_data[['Customer ID', 'Predicted Churn', 'Churn Probability'] + features])

            # Calculate cluster-wise statistics
            cluster_stats = city_data.groupby('Cluster').agg({
                'Churn Probability': 'mean',
                'Predicted Churn': lambda x: (x == 'Yes').mean(),
                'Age': 'mean',
                'Tenure in Months': 'mean',
                'Monthly Charge': 'mean',
                'Total Charges': 'mean'
            }).reset_index()

            cluster_stats.columns = ['Cluster', 'Avg Churn Probability', 'Churn Rate', 'Avg Age', 'Avg Tenure', 'Avg Monthly Charge', 'Avg Total Charges']

            # Calculate the percentage of users in each cluster
            cluster_percentages = city_data['Cluster'].value_counts(normalize=True) * 100
            cluster_stats['Percentage of Users'] = cluster_stats['Cluster'].map(cluster_percentages)

            st.write("Cluster-wise Statistics")
            st.dataframe(cluster_stats)

            # Generate cluster-wise recommendations
            cluster_info = cluster_stats.to_dict(orient='records')
            prompt = (
                f"Cluster-wise statistics for {selected_city}:\n{cluster_info}\n\n"
                "Based on these statistics, provide a general description and recommendations for each cluster. "
                "Consider the percentage of users in each cluster when making recommendations. Only include relevant clusters "
                "(e.g., if there is only 3% of customers in a cluster, it might not be necessary to provide detailed recommendations for that cluster)."
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a customer service expert providing recommendations to reduce churn based on cluster statistics."},
                    {"role": "user", "content": prompt}
                ]
            )

            cluster_recommendations = response.choices[0].message.content.strip()
            st.write('Cluster Recommendations:', cluster_recommendations)

            # Plot clusters
            fig = px.scatter(city_data, x='Longitude', y='Latitude', color='Cluster', hover_data=['Customer ID', 'Predicted Churn', 'Churn Probability'])
            st.plotly_chart(fig)

            csv = city_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download city prediction data as CSV",
                data=csv,
                file_name=f'{selected_city}_predictions.csv',
                mime='text/csv',
            )
        else:
            st.write("No customer data found for the selected city.")
