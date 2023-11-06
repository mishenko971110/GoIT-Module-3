import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

url = 'https://uk.wikipedia.org/wiki/%D0%9D%D0%B0%D1%81%D0%B5%D0%BB%D0%B5%D0%BD%D0%BD%D1%8F_%D0%A3%D0%BA%D1%80%D0%B0%D1%97%D0%BD%D0%B8'
re_caption = r'^Коефіцієнт народжуваності в регіонах України \(1950\—2019\)*$'
table_coef_birth = pd.read_html(url, match=re.compile(re_caption, re.I), thousands='', decimal=',')

#Виведення початкової таблиці
data_first = pd.DataFrame(table_coef_birth[0])
print(data_first)

#Вивести перші рядки таблиці за допомогою методу head
print(data_first.head())

#Визначте кількість рядків та стовпців у датафреймі
rows, cols = data_first.shape
print(data_first.shape)

#Замініть у таблиці значення "—" на значення NaN
data_first.replace('—', np.nan, inplace=True)
print(data_first)

#Визначте типи всіх стовпців за допомогою dataframe.dtypes
cols_types = data_first.dtypes
print(cols_types)

#Замініть типи нечислових колонок на числові. Підказка - це колонки, де знаходився символ "—"
object_cols = cols_types[cols_types == 'object'].index.tolist()
for i in object_cols[1:]:
    data_first[i] = data_first[i].astype(float)
print(data_first.dtypes)

#Порахуйте, яка частка пропусків міститься в кожній колонці (використовуйте методи isnull та sum)
print(data_first.isna().sum()/rows)

#Видаліть з таблиці дані по всій країні, останній рядок таблиці
modified_data = data_first.drop([rows-1])
print(modified_data)

#Замініть відсутні дані в стовпцях середніми значеннями цих стовпців (метод fillna)
modified_data.fillna(np.round(modified_data.mean(axis=0, numeric_only=True), 1), inplace=True)
print(modified_data)

#Отримайте список регіонів, де рівень народжуваності у 2019 році був вищим за середній по Україні
avg_value = np.round(modified_data['2019'].mean(axis=0, numeric_only=True), 2)
print("Середнє: ", avg_value)
avg_value_index = modified_data[modified_data['2019'] > avg_value].index.tolist()
avg_value_regions_list = modified_data['Регіон'].iloc[avg_value_index].tolist()
print(avg_value_regions_list)

#У якому регіоні була найвища народжуваність у 2014 році?
max_value = modified_data['2014'].max(axis=0, numeric_only=True)
max_value_index = modified_data[modified_data['2014'] == max_value].index.tolist()
max_value_region = modified_data['Регіон'].iloc[max_value_index]
print(max_value_region)

#Побудуйте стовпчикову діаграму народжуваності по регіонах у 2019 році
birth_rates_2019 = modified_data['2019'].values
regions = modified_data['Регіон'].tolist()

plt.figure(figsize=(10,10))
plt.bar(regions, birth_rates_2019)
plt.xlabel('Регіон')
plt.ylabel('Коефіцієнт народжуваності')
plt.title('Коефіцієнт народжуваності по регіонах у 2019 році')
plt.xticks(rotation=45)
plt.show()