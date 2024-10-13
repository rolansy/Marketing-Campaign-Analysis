# Re-importing the necessary library and running the code again
import pandas as pd

# Traditional Campaign Dataset
traditional_data = {
    'MarketID': [1, 2, 3, 4, 5],
    'MarketSize': ['Small', 'Medium', 'Large', 'Small', 'Large'],
    'LocationID': [101, 102, 103, 104, 105],
    'AgeOfStore': [5, 10, 15, 8, 12],
    'Promotion': [1, 2, 3, 1, 2],
    'Week': [1, 1, 2, 2, 3],
    'SalesInThousands': [10.5, 15.2, 20.1, 14.8, 19.3]
}

# Online Campaign Dataset (Facebook)
online_data = {
    'impressions': [10000, 15000, 20000, 12000, 18000],
    'clicks': [200, 300, 400, 250, 350],
    'CPC': [0.5, 0.6, 0.7, 0.55, 0.65],
    'CPM': [10, 12, 14, 11, 13],
    'reach': [8000, 12000, 15000, 9000, 14000],
    'conversions': [10, 15, 20, 12, 18],
    'ctr': [0.02, 0.02, 0.02, 0.02, 0.02]  # click-through rate
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
