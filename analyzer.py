import nbformat as nbf

# Creating a new Jupyter notebook
nb = nbf.v4.new_notebook()

# Adding markdown cells and code cells with the provided step-by-step code

# Markdown cell 1: Title
nb['cells'].append(nbf.v4.new_markdown_cell("# Marketing Campaign Data Analytics"))

# Code cell 1: Import libraries
code_cell_1 = '''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
'''
nb['cells'].append(nbf.v4.new_code_cell(code_cell_1))

# Code cell 2: Load datasets
code_cell_2 = '''
# Load Traditional and Online Campaign Datasets
traditional_df = pd.read_csv('traditional_campaign.csv')  # Use appropriate path
online_df = pd.read_csv('facebook_campaign.csv')

# Display first few rows
print(traditional_df.head())
print(online_df.head())
'''
nb['cells'].append(nbf.v4.new_code_cell(code_cell_2))

# Code cell 3: Data Cleaning
code_cell_3 = '''
# Handling missing values in traditional data (if any)
traditional_df.isna().sum()

# Cleaning online campaign data
online_df.isna().sum()

# Fill missing values as described in the paper
online_df.fillna(0, inplace=True)  # You can modify this based on specific columns
'''
nb['cells'].append(nbf.v4.new_code_cell(code_cell_3))

# Code cell 4: EDA
code_cell_4 = '''
# Plot sales by promotion type
sns.boxplot(x='Promotion', y='SalesInThousands', data=traditional_df)
plt.title('Sales by Promotion Type')
plt.show()

# Plot for Facebook dataset - Impressions vs Clicks
sns.scatterplot(x='impressions', y='clicks', data=online_df)
plt.title('Impressions vs Clicks')
plt.show()
'''
nb['cells'].append(nbf.v4.new_code_cell(code_cell_4))

# Code cell 5: Feature selection and preprocessing
code_cell_5 = '''
# Traditional Campaign
X_traditional = traditional_df[['MarketSize', 'AgeOfStore', 'Promotion', 'Week']]
y_traditional = traditional_df['SalesInThousands']

# Online Campaign
X_online = online_df[['impressions', 'clicks', 'CPC', 'reach']]
y_online = online_df['ctr']  # Click-through rate as target

# Splitting into train-test sets
X_train_trad, X_test_trad, y_train_trad, y_test_trad = train_test_split(X_traditional, y_traditional, test_size=0.3, random_state=42)
X_train_online, X_test_online, y_train_online, y_test_online = train_test_split(X_online, y_online, test_size=0.3, random_state=42)
'''
nb['cells'].append(nbf.v4.new_code_cell(code_cell_5))

# Code cell 6: Model training
code_cell_6 = '''
# Random Forest for Traditional Data
rf_model_trad = RandomForestRegressor()
rf_model_trad.fit(X_train_trad, y_train_trad)

# Random Forest for Online Data
rf_model_online = RandomForestRegressor()
rf_model_online.fit(X_train_online, y_train_online)
'''
nb['cells'].append(nbf.v4.new_code_cell(code_cell_6))

# Code cell 7: Model evaluation
code_cell_7 = '''
# Predictions for traditional data
y_pred_trad = rf_model_trad.predict(X_test_trad)

# Predictions for online data
y_pred_online = rf_model_online.predict(X_test_online)

# Calculate RMSE
rmse_trad = np.sqrt(metrics.mean_squared_error(y_test_trad, y_pred_trad))
rmse_online = np.sqrt(metrics.mean_squared_error(y_test_online, y_pred_online))

print(f'RMSE for Traditional Campaign: {rmse_trad}')
print(f'RMSE for Online Campaign: {rmse_online}')
'''
nb['cells'].append(nbf.v4.new_code_cell(code_cell_7))

# Code cell 8: Results visualization
code_cell_8 = '''
# Actual vs Predicted for Traditional Campaign
plt.scatter(y_test_trad, y_pred_trad)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Traditional Campaign - Actual vs Predicted')
plt.show()

# Actual vs Predicted for Online Campaign
plt.scatter(y_test_online, y_pred_online)
plt.xlabel('Actual CTR')
plt.ylabel('Predicted CTR')
plt.title('Online Campaign - Actual vs Predicted')
plt.show()
'''
nb['cells'].append(nbf.v4.new_code_cell(code_cell_8))

# Save the notebook as .ipynb file
notebook_filename = "Marketing_Campaign_Analytics.ipynb"
with open(notebook_filename, 'w') as f:
    nbf.write(nb, f)

notebook_filename
