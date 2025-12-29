# Файл: create_hr_data.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
# Створити 50 записів про кандидатів
data = {
    'age': np.random.randint(22, 56, 50),           # Вік 22-55
    'salary_expectation': np.random.randint(500, 5001, 50),  # Очікування зп 500-5000
    'experience': np.random.randint(0, 21, 50),     # Досвід 0-20 років
    'education_level': np.random.choice([1, 2, 3], 50, p=[0.3, 0.5, 0.2]),  # 1=середня, 2=бакалавр, 3=магістр
    'english_level': np.random.choice([1, 2, 3, 4], 50, p=[0.2, 0.3, 0.3, 0.2]),  # 1=A1, 4=C1
}

df = pd.DataFrame(data)

# Додати цільову змінну: чи наймаємо? (1=так, 0=ні)
# Логіка: наймаємо якщо (досвід > 3 І очікування зп < 3000) АБО (рівень англійської >= 3)
df['hired'] = ((df['experience'] > 3) & (df['salary_expectation'] < 3000)) | (df['english_level'] >= 3)
df['hired'] = df['hired'].astype(int)

# Зберегти
df.to_csv('data/hr_candidates.csv', index=False)
print("✅ Створено датасет з 50 кандидатами")
print(df.head(30))
x = df[["age", "salary_expectation", "experience", "education_level", "english_level"]]  # ПРАВИЛЬНО
y=df["hired"]

model = DecisionTreeClassifier()
model.fit(x, y)
from sklearn.tree import export_text
tree_rules = export_text(
    model,
    feature_names=["age", "salary_expectation", "experience", "education_level", "english_level"] # ПРАВИЛЬНО
)
print(tree_rules)