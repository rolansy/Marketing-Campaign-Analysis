# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Data
facebook_df = pd.read_csv('facebook_campaign.csv')
traditional_df = pd.read_csv('traditional_campaign.csv')

# Step 3: Data Preprocessing
# Convert categorical variables to numeric for traditional campaign
traditional_df['MarketSize'] = traditional_df['MarketSize'].map({'Small': 1, 'Medium': 2, 'Large': 3})

# Check for missing values
print(facebook_df.isnull().sum())
print(traditional_df.isnull().sum())

# Verify column names
print(facebook_df.columns)
print(traditional_df.columns)

# Step 4: Exploratory Data Analysis (EDA)
# Facebook Campaign Data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='impressions', y='clicks', data=facebook_df)
plt.title('Impressions vs Clicks (Facebook Campaign)')
plt.xlabel('Impressions')
plt.ylabel('Clicks')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='CPC', y='conversions', data=facebook_df)
plt.title('CPC vs Conversions (Facebook Campaign)')
plt.xlabel('CPC')
plt.ylabel('Conversions')
plt.show()

# Traditional Campaign Data
plt.figure(figsize=(10, 6))
sns.boxplot(x='MarketSize', y='SalesInThousands', hue='Promotion', data=traditional_df)
plt.title('Sales by Market Size and Promotion (Traditional Campaign)')
plt.xlabel('Market Size')
plt.ylabel('Sales in Thousands')
plt.legend(title='Promotion')
plt.show()

# Step 5: Feature Selection and Scaling
# Facebook Campaign
X_facebook = facebook_df[['impressions', 'clicks', 'CPC', 'CPM', 'reach', 'conversions']]
y_facebook = facebook_df['ctr']

# Traditional Campaign
X_traditional = traditional_df[['MarketSize', 'AgeOfStore', 'Promotion', 'Week']]
y_traditional = traditional_df['SalesInThousands']

# Scaling
scaler = StandardScaler()
X_facebook_scaled = scaler.fit_transform(X_facebook)
X_traditional_scaled = scaler.fit_transform(X_traditional)

# Step 6: Model Training and Prediction
# Split data into training and testing sets
X_train_fb, X_test_fb, y_train_fb, y_test_fb = train_test_split(X_facebook_scaled, y_facebook, test_size=0.2, random_state=42)
X_train_trad, X_test_trad, y_train_trad, y_test_trad = train_test_split(X_traditional_scaled, y_traditional, test_size=0.2, random_state=42)

# Initialize KNN Regressor
knn_fb = KNeighborsRegressor(n_neighbors=5)
knn_trad = KNeighborsRegressor(n_neighbors=5)

# Train the models
knn_fb.fit(X_train_fb, y_train_fb)
knn_trad.fit(X_train_trad, y_train_trad)

# Make predictions
y_pred_fb = knn_fb.predict(X_test_fb)
y_pred_trad = knn_trad.predict(X_test_trad)

# Step 7: Visualization and Conclusion
# Facebook Campaign Predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test_fb, y_pred_fb)
plt.plot([0, 0.03], [0, 0.03], 'r--')
plt.title('Actual vs Predicted CTR (Facebook Campaign)')
plt.xlabel('Actual CTR')
plt.ylabel('Predicted CTR')
plt.show()

# Traditional Campaign Predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test_trad, y_pred_trad)
plt.plot([10, 25], [10, 25], 'r--')
plt.title('Actual vs Predicted Sales (Traditional Campaign)')
plt.xlabel('Actual Sales in Thousands')
plt.ylabel('Predicted Sales in Thousands')
plt.show()

# Evaluation Metrics
mse_fb = mean_squared_error(y_test_fb, y_pred_fb)
r2_fb = r2_score(y_test_fb, y_pred_fb)
mse_trad = mean_squared_error(y_test_trad, y_pred_trad)
r2_trad = r2_score(y_test_trad, y_pred_trad)

print(f'Facebook Campaign - MSE: {mse_fb}, R²: {r2_fb}')
print(f'Traditional Campaign - MSE: {mse_trad}, R²: {r2_trad}')

# Conclusion
print("""
### Conclusion
- The KNN model was used to predict CTR for the Facebook campaign and sales for the traditional campaign.
- The scatter plots show the relationship between actual and predicted values.
- The evaluation metrics (MSE and R²) indicate the performance of the models.
- Further tuning and feature engineering may improve model performance.
""")