import urllib.request
import os

# The public URL for the Telco dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Where we want to save it
save_path = os.path.join("ml_engine", "data", "Telco-Customer-Churn.csv")

print("Downloading dataset...")
urllib.request.urlretrieve(url, save_path)
print(f"Success! Dataset saved to: {save_path}")