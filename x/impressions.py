import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('impressions_model.pkl')

# Load the dataset for analysis
data = pd.read_csv('data.csv')

# Convert categorical variables to numeric
data['gender'] = data['gender'].map({'M': 0, 'F': 1})

# Convert age to numeric by taking the average of the range and converting to integer
def convert_age(age):
    if isinstance(age, str):
        return int(np.mean(list(map(int, age.split('-')))))
    return np.nan

data['age'] = data['age'].apply(convert_age)

# Drop rows with missing values in 'age' column
data.dropna(subset=['age'], inplace=True)

# Ensure all other columns are numeric
for col in ['ad_id', 'campaign_id', 'fb_campaign_id', 'interest1', 'interest2', 'interest3', 'clicks', 'spent', 'total_conversion', 'approved_conversion']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any remaining missing values
data.dropna(inplace=True)

# Define the input fields
st.title('Impressions Predictor')

if st.button('Generate Random Values'):
    ad_id = np.random.randint(100000, 999999)
    campaign_id = np.random.randint(1000, 9999)
    fb_campaign_id = np.random.randint(100000, 999999)
else:
    ad_id = st.number_input('Ad ID', min_value=0)
    campaign_id = st.number_input('Campaign ID', min_value=0)
    fb_campaign_id = st.number_input('FB Campaign ID', min_value=0)

age = st.selectbox('Age', ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
gender = st.selectbox('Gender', ['M', 'F'])
interest1 = st.slider('Interest 1', min_value=0, max_value=100)
interest2 = st.slider('Interest 2', min_value=0, max_value=100)
interest3 = st.slider('Interest 3', min_value=0, max_value=100)
clicks = st.slider('Clicks', min_value=0, max_value=100)
spent = st.slider('Spent', min_value=0.0, max_value=1000.0, format="%.2f")
total_conversion = st.slider('Total Conversion', min_value=0, max_value=100)
approved_conversion = st.slider('Approved Conversion', min_value=0, max_value=100)

# Convert age to numeric by taking the average of the range and converting to integer
age = convert_age(age)

# Convert gender to numeric
gender = 0 if gender == 'M' else 1

# Create a feature array
features = np.array([[ad_id, campaign_id, fb_campaign_id, age, gender, interest1, interest2, interest3, clicks, spent, total_conversion, approved_conversion]])

# Predict impressions
if st.button('Predict Impressions'):
    prediction = model.predict(features)
    st.write(f'Predicted Impressions: {int(prediction[0])}')

# Add interactive graphs
st.subheader('Data Analysis')

# Plot distribution of impressions
st.write('### Distribution of Impressions')
fig, ax = plt.subplots()
sns.histplot(data['impressions'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Plot impressions vs. clicks
st.write('### Impressions vs. Clicks')
fig, ax = plt.subplots()
sns.scatterplot(x=data['clicks'], y=data['impressions'], ax=ax)
st.pyplot(fig)

# Plot impressions vs. spent
st.write('### Impressions vs. Spent')
fig, ax = plt.subplots()
sns.scatterplot(x=data['spent'], y=data['impressions'], ax=ax)
st.pyplot(fig)

# Explanation based on the Facebook marketing campaign data
st.write("""
### Explanation
This application predicts the number of impressions for a Facebook marketing campaign based on various input features such as age, gender, interests, clicks, spent amount, and conversions. The model was trained on historical campaign data to understand the relationship between these features and the resulting impressions.
""")