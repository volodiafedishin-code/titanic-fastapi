import pandas as pd
import numpy as np

# Створимо великий CSV файл (100K рядків)
np.random.seed(42)
n_rows = 1000

data = {
    'order_id': [f'ORD_{i:06d}' for i in range(1, n_rows + 1)],
    'customer_id': [f'CUST_{np.random.randint(1000, 9999)}' for _ in range(n_rows)],
    'product_id': np.random.choice([f'P{str(i).zfill(3)}' for i in range(1, 51)], n_rows),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'price': np.random.uniform(10, 1000, n_rows).round(2),
    'quantity': np.random.randint(1, 10, n_rows),
    'order_date': pd.date_range('2023-01-01', periods=n_rows, freq='T'),
    'country': np.random.choice(['UA', 'PL', 'DE', 'US', 'UK'], n_rows, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
    'discount': np.random.choice([0, 5, 10, 15], n_rows, p=[0.6, 0.2, 0.15, 0.05]),
    'shipping_cost': np.random.choice([0, 5, 10, 20], n_rows)
}

df = pd.DataFrame(data)
df.to_csv('data/ecommerce_large.csv', index=False)
print(f"Файл створено: data/ecommerce_large.csv ({len(df):,} рядків)")
print(df)