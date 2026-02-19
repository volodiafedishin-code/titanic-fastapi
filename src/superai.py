import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Завантаження даних
# Припустимо, файл titanic.csv лежить у тій же папці
df = pd.read_csv(r"C:\Users\volod\ml_engineer\ml_basics\data\tytanic.csv")

print(df)
# Для прикладу створимо змінні X та y вручну
X = df[['Age', 'Fare', 'Pclass']]
y = df['Survived']

# 2. Розділення на Train/Test (30% на перевірку)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Створення "розумного" конвеєра (Pipeline)
full_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # Заповнюємо пусті клітинки середнім
    ('scaler', StandardScaler()),                # Приводимо до одного масштабу
    ('model', RandomForestClassifier(n_estimators=100, random_state=42)) # Модель
])

# 4. Навчання (весь процес обробки та навчання запускається однією командою)
full_pipeline.fit(X_train, y_train)

# 5. Оцінка результатів
predictions = full_pipeline.predict(X_test)
print("Звіт про якість моделі:")
print(classification_report(y_test, predictions))

# 6. Збереження результату у файл (артефакт моделі)
joblib.dump(full_pipeline, r"C:\Users\volod\ml_engineer\ml_basics\models\titanic_model.pkl")
print("\nМодель збережена як 'titanic_model.pkl'. Роботу завершено! ✅")