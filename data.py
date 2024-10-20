# Re-importing the necessary library and running the code again
import pandas as pd

# Expanded Traditional Campaign Dataset
traditional_data = {
    'MarketID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'MarketSize': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Small', 'Large', 'Medium', 'Small', 'Large', 'Small', 'Medium', 'Large', 'Medium'],
    'LocationID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    'AgeOfStore': [5, 10, 15, 8, 12, 7, 6, 14, 13, 9, 16, 11, 8, 10, 12],
    'Promotion': [1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 3, 2, 1],
    'Week': [1, 1, 2, 2, 3, 4, 5, 6, 3, 4, 2, 1, 5, 6, 4],
    'SalesInThousands': [10.5, 15.2, 20.1, 14.8, 19.3, 18.5, 11.3, 23.1, 16.4, 13.8, 22.5, 17.9, 15.0, 21.2, 16.7]
}


# Expanded Online Campaign Dataset (Facebook)
online_data = {
    'impressions': [10000, 15000, 20000, 12000, 18000, 16000, 25000, 30000, 21000, 19000, 17000, 26000, 22000, 28000, 24000],
    'clicks': [200, 300, 400, 250, 350, 300, 500, 600, 450, 380, 320, 530, 420, 560, 480],
    'CPC': [0.5, 0.6, 0.7, 0.55, 0.65, 0.62, 0.58, 0.72, 0.63, 0.68, 0.6, 0.7, 0.67, 0.73, 0.66],
    'CPM': [10, 12, 14, 11, 13, 12.5, 13.8, 14.2, 12.9, 13.3, 11.7, 13.5, 12.1, 14.8, 13.4],
    'reach': [8000, 12000, 15000, 9000, 14000, 13000, 17000, 20000, 16000, 15000, 14000, 18000, 15000, 19000, 17000],
    'conversions': [10, 15, 20, 12, 18, 16, 25, 30, 22, 19, 17, 26, 21, 28, 24],
    'ctr': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
}


# Convert dictionaries to dataframes
traditional_df = pd.DataFrame(traditional_data)
online_df = pd.DataFrame(online_data)

# Save the dataframes as CSV files
traditional_csv_path = "traditional_campaign.csv"
online_csv_path = "facebook_campaign.csv"

traditional_df.to_csv(traditional_csv_path, index=False)
online_df.to_csv(online_csv_path, index=False)

# Return paths of CSV files
traditional_csv_path, online_csv_path
