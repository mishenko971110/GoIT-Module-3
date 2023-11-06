import pandas as pd
import numpy as np

#Прочитайте файл 2017_jun_final.csv за допомогою методу read_csv
data = pd.read_csv("2017_jun_final.csv")
print(data)

#Прочитайте отриману таблицю, використовуючи метод head
print(data.head())

#Визначте розмір таблиці за допомогою методу shape
rows, cols = data.shape
print(rows, cols)

#Визначте типи всіх стовпців за допомогою dataframe.dtypes
cols_types = data.dtypes
print(cols_types)

#Порахуйте, яка частка пропусків міститься в кожній колонці (використовуйте методи isnull та sum)
print(data.isna().sum()/rows)

#Видаліть усі стовпці з пропусками, крім стовпця "Мова програмування"
except_col = 'Язык.программирования'
keys_list = []
for key in data.keys():
    if data[key].isna().sum()/rows != 0 and key != except_col:
        keys_list.append(key)
print(keys_list)

modified_data = data.drop(keys_list, axis=1)
print(modified_data)

#Знову порахуйте, яка частка пропусків міститься в кожній колонці і переконайтеся, що залишився тільки стовпець "Мова.програмування"
print(modified_data.isna().sum()/rows)

#Видаліть усі рядки у вихідній таблиці за допомогою методу dropna
modified_data = modified_data.dropna()
print(modified_data)

#Визначте новий розмір таблиці за допомогою методу shape
print(modified_data.shape)

#Створіть нову таблицю python_data, в якій будуть тільки рядки зі спеціалістами, які вказали мову програмування Python
python_data = modified_data[modified_data['Язык.программирования'] == 'Python']
print(python_data)

#Визначте розмір таблиці python_data за допомогою методу shape
print(python_data.shape)

#Використовуючи метод groupby, виконайте групування за стовпчиком "Посада"
#Створіть новий DataFrame, де для згрупованих даних за стовпчиком "Посада", 
#виконайте агрегацію даних за допомогою методу agg і знайдіть мінімальне та максимальне значення у стовпчику "Зарплата.в.місяць"
grouped_python_data = python_data.groupby(by = ['Должность'], axis=0, sort=True).agg(min=('Зарплата.в.месяц', 'min'), max=('Зарплата.в.месяц', 'max'))
print(grouped_python_data)

#Створіть функцію fill_avg_salary, яка повертатиме середнє значення заробітної плати на місяць. 
#Використовуйте її для методу apply та створіть новий стовпчик "avg"
def fill_avg_salary(row):
    return (row['min'] + row['max']) / 2

grouped_python_data['avg'] = grouped_python_data.apply(fill_avg_salary, axis=1)
print(grouped_python_data)

#Створіть описову статистику за допомогою методу describe для нового стовпчика.
grouped_python_data['avg'].describe()

#Збережіть отриману таблицю в CSV файл
grouped_python_data.to_csv('output_data.csv')
