import pandas as pd

# 1. Створюємо словник з даними (додайте сюди 5 значень для кожного ключа)
data = {
    'salary': [1200, 4800, 5200, 1100, 4600],
    'experience': [1, 6, 8, 2, 5],
    'age': [19, 32, 45, 21, 28]
}

# 2. Перетворюємо на DataFrame
df = pd.DataFrame(data)

# 3. ТУТ ВАШ РЯДОК: створіть колонку 'is_young'
# Підказка: використовуйте (df['age'] < 23).astype(int)
df["is young"]=(df["age"]>23).astype(int)
print(df)