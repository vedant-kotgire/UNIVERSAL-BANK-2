import pandas as pd
import numpy as np

np.random.seed(42)
n = 50

data = {
    'ID': range(5001, 5001 + n),
    'Age': np.random.randint(23, 68, n),
    'Experience': np.random.randint(0, 44, n),
    'Income': np.random.randint(8, 225, n),
    'ZIP Code': np.random.randint(90000, 97000, n),
    'Family': np.random.randint(1, 5, n),
    'CCAvg': np.round(np.random.uniform(0, 10, n), 1),
    'Education': np.random.randint(1, 4, n),
    'Mortgage': np.random.choice([0, 0, 0, 50, 100, 150, 200, 300], n),
    'Securities Account': np.random.randint(0, 2, n),
    'CD Account': np.random.randint(0, 2, n),
    'Online': np.random.randint(0, 2, n),
    'CreditCard': np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)
df.to_csv('data/Test_UniversalBank.csv', index=False)
print(f"Test file created with {len(df)} rows")
print(df.head())
