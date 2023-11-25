import pandas as pd

#Прочитайте csv файл (використовуйте функцію read_csv)
books = pd.read_csv("bestsellers with categories.csv")
print(books)

#Виведіть перші п'ять рядків (використовується функція head)
print(books.head())

#Виведіть розміри датасету (використовуйте атрибут shape)
rows, cols = books.shape
print(rows, cols)
book_count = books['Name'].nunique()
print("Про скільки книг зберігає дані датасет? - ", book_count)

#Давайте змінимо регістр на малий, а пробіл замінимо на нижнє підкреслення (snake_style). 
#А заразом і вивчимо корисний атрибут датафрейму: columns (можна просто присвоїти список нових імен цьому атрибуту)
books.columns = ['name', 'author', 'user_rating', 'reviews', 'price', 'year', 'genre']
print(books)

#Перевірте, чи у всіх рядків вистачає даних: виведіть кількість пропусків (na) у кожному зі стовпців (використовуйте функції isna та sum)
print(books.isna().sum())

#Відповідь: Чи є в якихось змінних пропуски? (Так / ні)
if books.isna().sum().sum() == 0:
    print('Чи є в якихось змінних пропуски? - Hi')
else:
    print('Чи є в якихось змінних пропуски? - Так')

#Відповідь: Які є унікальні жанри?
print('Які є унікальні жанри? - ', books['genre'].unique())

#Тепер подивіться на розподіл цін: побудуйте діаграму (використовуйте kind='hist')
books['price'].plot.hist(bins=10, alpha=1)

#Відповідь: Максимальна ціна?
print('Максимальна ціна? - ', books['price'].max())

#Відповідь: Мінімальна ціна?
print('Мінімальна ціна? - ', books['price'].min())

#Відповідь: Середня ціна?
print('Середня ціна? - ', books['price'].mean())

#Відповідь: Медіанна ціна?
print('Медіанна ціна? - ', books['price'].median())

#Відповідь: Який рейтинг у датасеті найвищий?
print('Який рейтинг у датасеті найвищий? - ', books['user_rating'].max())

#Відповідь: Скільки книг мають такий рейтинг?
print('Скільки книг мають такий рейтинг? - ', books['user_rating'][books['user_rating'] == books['user_rating'].max()].count())

#Відповідь: Яка книга має найбільше відгуків?
print('Яка книга має найбільше відгуків? - ', books['name'][books['reviews'] == books['reviews'].max()].item())

#Відповідь: З тих книг, що потрапили до Топ-50 у 2015 році, яка книга найдорожча (можна використати проміжний датафрейм)?
books_2015 = books[books['year'] == 2015]
books_2015_max_price = books_2015['name'][books_2015['price'] == books_2015['price'].max()]
print('З тих книг, що потрапили до Топ-50 у 2015 році, яка книга найдорожча? - ', books_2015_max_price.item())

#Відповідь: Скільки книг жанру Fiction потрапили до Топ-50 у 2010 році (використовуйте &)?
print('Скільки книг жанру Fiction потрапили до Топ-50 у 2010 році (використовуйте &)? - ', books['name'][(books['year'] == 2010) & (books['genre'] == 'Fiction')].count())

#Відповідь: Скільки книг з рейтингом 4.9 потрапило до рейтингу у 2010 та 2011 роках (використовуйте | або функцію isin)?
print('Скільки книг з рейтингом 4.9 потрапило до рейтингу у 2010 та 2011 роках? - ', books['name'][((books['year'] == 2010) | (books['year'] == 2011)) & (books['user_rating'] == 4.9)].count())

#І насамкінець, давайте відсортуємо за зростанням ціни всі книги, які потрапили до рейтингу в 2015 році і коштують дешевше за 8 доларів (використовуйте функцію sort_values).
#Відповідь: Яка книга остання у відсортованому списку?
sorted_filtered_books = books[(books['year'] == 2015) & (books['price'] < 8)].sort_values(by = ['price'])
print('Яка книга остання у відсортованому списку?', sorted_filtered_books['name'].tail(1).item())

#Відповідь: Максимальна ціна для жанру Fiction
print('Максимальна ціна для жанру Fiction? - ', books['price'][books['genre'] == 'Fiction'].agg('max'))

#Відповідь: Мінімальна ціна для жанру Fiction
print('Мінімальна ціна для жанру Fiction? - ', books['price'][books['genre'] == 'Fiction'].agg('min'))

#Відповідь: Максимальна ціна для жанру Non Fiction
print('Максимальна ціна для жанру Non Fiction? - ', books['price'][books['genre'] == 'Non Fiction'].agg('max'))

#Відповідь: Мінімальна ціна для жанру Non Fiction
print('Мінімальна ціна для жанру Non Fiction? - ', books['price'][books['genre'] == 'Non Fiction'].agg('min'))

#Тепер створіть новий датафрейм, який вміщатиме кількість книг для кожного з авторів 
#(використовуйте функції groupby та agg, для підрахунку кількості використовуйте count).
author_books_count = books.groupby('author').agg({'name': 'count'}).reset_index()
author_books_count.columns = ['author', 'book_count']
print('Cтворіть новий датафрейм, який вміщатиме кількість книг для кожного з авторів: \n', author_books_count.sort_values(by='book_count', ascending=False))

#Відповідь: Якої розмірності вийшла таблиця?
print('Якої розмірності вийшла таблиця? - ', author_books_count.shape)

#Відповідь: Який автор має найбільше книг?
max_author_name = author_books_count['author'][author_books_count['book_count'] == author_books_count['book_count'].max()].item()
print('Який автор має найбільше книг? - ', max_author_name)

#Відповідь: Скільки книг цього автора?
print('Скільки книг цього автора? - ', author_books_count['book_count'][author_books_count['author'] == max_author_name].item())

#Тепер створіть другий датафрейм, який буде вміщати середній рейтинг для кожного автора 
#(використовуйте функції groupby та agg, для підрахунку середнього значення використовуйте mean). 
author_rating_avg = books.groupby('author').agg({'user_rating': 'mean'}).reset_index()
author_rating_avg.columns = ['author', 'avg_rating']
print('Cтворіть новий датафрейм, який буде вміщати середній рейтинг для кожного автора: \n', author_rating_avg.sort_values(by='avg_rating', ascending=False))

#Відповідь: У якого автора середній рейтинг мінімальний?
min_rating_author = author_rating_avg['author'][author_rating_avg['avg_rating'] == author_rating_avg['avg_rating'].min()].item()
print('У якого автора середній рейтинг мінімальний? - ', min_rating_author)

#Відповідь: Який у цього автора середній рейтинг?
print('Який у цього автора середній рейтинг? - ', author_rating_avg['avg_rating'][author_rating_avg['author'] == min_rating_author].item())

#З'єднайте останні два датафрейми так, щоб для кожного автора було видно кількість книг та середній рейтинг
result_frame = pd.concat([author_books_count, author_rating_avg['avg_rating']], axis=1)
print('З\'єднайте останні два датафрейми так, щоб для кожного автора було видно кількість книг та середній рейтинг: \n', result_frame)

#Відсортуйте датафрейм за зростаючою кількістю книг та зростаючим рейтингом (використовуйте функцію sort_values)
sort_result_frame = result_frame.sort_values(by=['book_count', 'avg_rating'], ascending=[True, True])
print(sort_result_frame)

#Відповідь: Який автор перший у списку?
print('Який автор перший у списку? - ', sort_result_frame.iloc[0]['author'])