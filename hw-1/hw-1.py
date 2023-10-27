import numpy as np
import math

def createRandVector(size, num_from, num_to):
    if num_to > 1:
        return np.random.randint(num_from, num_to, size)
    elif num_from == 0 and num_to == 0:
        return np.zeros(size, dtype=int)
    elif num_from == 1 and num_to == 1:
        return np.ones(size, dtype=int)
    else:
        return np.random.random(size)
    
def createNotRandVector(num_from, num_to):
    return np.linspace(num_from, num_to, num=num_to, dtype=int)
    
def createMatrix(h, w, num_from, num_to):
    if num_from == 0 and num_to == 0:
        return np.zeros((h, w))
    elif num_from == 1 and num_to == 1:
        return np.ones((h, w))
    elif num_to > 1:
        return np.random.randint(num_from, num_to + 1, size=(h, w))
    else:
        return np.random.random((h, w))

def matrixSum(matrix_1, matrix_2):
    return np.add(matrix_1, matrix_2)

def matrixSub(matrix_1, matrix_2):
    return np.subtract(matrix_1, matrix_2)

def matrixMul(matrix_1, matrix_2):
    return np.multiply(matrix_1, matrix_2)

def matrixDiv(matrix_1, matrix_2):
    return np.divide(matrix_1, matrix_2)

def matrixInv(matrix):
    try:
        inverse_matrix = np.linalg.inv(matrix)
        return inverse_matrix
    except np.linalg.LinAlgError:
        print("\nЦя матриця не має оберненої матриці.")

def matrixTransp(matrix):
    return np.transpose(matrix)

def matrixSumNums(matrix):
    return matrix.sum()

def matrixSumNumsLine(matrix):
    matrixVector = []
    for i in range(len(matrix)):
        matrixVector.append(matrix[i].sum())
    return matrixVector

def matrixDoubleNums(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            temp = matrix[i][j]
            matrix[i][j] = temp*temp
    return matrix

def vectorDot(vect1, vect2):
    return np.dot(vect1, vect2)

def vectorSquardLine(vect):
    new_vect = []
    for i in range(len(vect)):
        new_vect.append(math.pow(vect[i], 1/2))
    return new_vect

def matrixMulVector(matrix, vect):
    return np.dot(matrix, vect)


def task1():
    print("\n-----\n#1-Створіть одновимірний масив (вектор) з першими 10-ма натуральними числами та виведіть його значення\n")
    vector = createNotRandVector(1, 10)
    print("Новий вектор: ", vector)

def task2():    
    print("\n-----\n#2-Створіть двовимірний масив (матрицю) розміром 3x3, заповніть його нулями та виведіть його значення\n")
    matrix = createMatrix(3, 3, 0, 0)
    print("Матриця: \n", matrix)

def task3():   
    print("\n-----\n#3-Створіть масив розміром 5x5, заповніть його випадковими цілими числами\n \
    в діапазоні від 1 до 10 та виведіть його значення\n")
    matrix = createMatrix(5, 5, 1, 10)
    print("Матриця: \n", matrix)

def task4():   
    print("\n-----\n#4-Створіть масив розміром 4x4, заповніть його випадковими дійсними числами\n \
    в діапазоні від 0 до 1 та виведіть його значення\n")
    matrix = createMatrix(4, 4, 0, 1)
    print("Матриця: \n", np.round(matrix, 2))

def task5():    
    print("\n-----\n#5-Створіть два одновимірних масиви розміром 5, заповніть їх випадковими цілими числами\n \
    в діапазоні від 1 до 10 та виконайте на них поелементні операції додавання, віднімання та множення\n")
    matrix1 = createMatrix(5, 5, 1, 10)
    matrix2 = createMatrix(5, 5, 1, 10)
    print("Матриця 1: \n", matrix1)
    print("Матриця 2: \n", matrix2)
    print("Додавання матриць: \n", matrixSum(matrix1, matrix2))
    print("Віднімання матриць: \n", matrixSub(matrix1, matrix2))
    print("Множення матриць: \n", matrixMul(matrix1, matrix2))

def task6():    
    print("\n-----\n#6-Створіть два вектори розміром 7, заповніть довільними числами \n \
    та знайдіть їх скалярний добуток\n")
    vect1 = createRandVector(7, 1, 50)
    vect2 = createRandVector(7, 1, 50)
    print("Вектор 1: ", vect1)
    print("Вектор 2: ", vect2)
    print("Скалярний добуток векторів: ", vectorDot(vect1, vect2))

def task7():   
    print("\n-----\n#7-Створіть дві матриці розміром 2x2 та 2x3, заповніть їх випадковими \n \
    цілими числами в діапазоні від 1 до 10 та перемножте їх між собою\n")
    matrix1 = createMatrix(2, 2, 1, 10)
    matrix2 = createMatrix(2, 3, 1, 10)
    print("Матриця 1: \n", matrix1)
    print("Матриця 2: \n", matrix2)
    print("Множення матриць: \n", matrixMulVector(matrix1, matrix2))

def task8():
    print("\n-----\n#8-Створіть матрицю розміром 3x3, заповніть її випадковими цілими числами \n \
    в діапазоні від 1 до 10 та знайдіть її обернену матрицю\n")
    matrix = createMatrix(3, 3, 1, 10)
    print("Матриця початкова: \n", matrix)
    print("Обернена матриця: \n", np.round(matrixInv(matrix), 2))

def task9():    
    print("\n-----\n#9-Створіть матрицю розміром 4x4, заповніть її випадковими дійсними числами \n \
    в діапазоні від 0 до 1 та транспонуйте її\n")
    matrix = createMatrix(4, 4, 0, 1)
    print("Матриця початкова: \n", np.round(matrix, 2))
    print("Транспонована матриця: \n", np.round(matrixTransp(matrix), 2))

def task10():    
    print("\n-----\n#10-Створіть матрицю розміром 3x4 та вектор розміром 4, заповніть їх \n \
    випадковими цілими числами в діапазоні від 1 до 10 та перемножте матрицю на вектор\n")
    matrix = createMatrix(3, 4, 1, 10)
    print("Матриця: \n", matrix)
    vect = createRandVector(4, 1, 10)
    print("Вектор: ", vect)
    print("Результат множення: ", matrixMulVector(matrix, vect))

def task11():   
    print("\n-----\n#11-Створіть матрицю розміром 2x3 та вектор розміром 3, заповніть їх випадковими \n \
    дійсними числами в діапазоні від 0 до 1 та перемножте матрицю на вектор\n")
    matrix = createMatrix(2, 3, 0, 1)
    print("Матриця: \n", np.round(matrix, 2))
    vect = createRandVector(3, 0, 1)
    print("Вектор: ", np.round(vect, 2))
    print("Результат множення: ", np.round(matrixMulVector(matrix, vect), 2))

def task12():    
    print("\n-----\n#12-Створіть дві матриці розміром 2x2, заповніть їх випадковими цілими числами в\n \
    діапазоні від 1 до 10 та виконайте їхнє поелементне множення\n")
    matrix1 = createMatrix(2, 2, 1, 10)
    matrix2 = createMatrix(2, 2, 1, 10)
    print("Матриця1: \n", matrix1)
    print("Матриця2: \n", matrix2)
    print("Множення матриць: \n", matrixMul(matrix1, matrix2))

def task13():   
    print("\n-----\n#13-Створіть дві матриці розміром 2x2, заповніть їх випадковими цілими числами в\n \
    діапазоні від 1 до 10 та знайдіть їх добуток\n")
    matrix1 = createMatrix(2, 2, 1, 10)
    matrix2 = createMatrix(2, 2, 1, 10)
    print("Матриця1: \n", matrix1)
    print("Матриця2: \n", matrix2)
    print("Множення матриць: \n", matrixMul(matrix1, matrix2))

def task14():    
    print("\n-----\n#14-Створіть матрицю розміром 5x5, заповніть її випадковими цілими числами в \n \
    діапазоні від 1 до 100 та знайдіть суму елементів матриці\n")
    matrix = createMatrix(5, 5, 1, 100)
    print("Матриця: \n", matrix)
    print("Сума елементів матриці: ", matrixSumNums(matrix))

def task15():
    print("\n-----\n#15-Створіть дві матриці розміром 4x4, заповніть їх випадковими цілими числами \n \
    в діапазоні від 1 до 10 та знайдіть їхню різницю\n")
    matrix1 = createMatrix(2, 3, 1, 10)
    matrix2 = createMatrix(2, 3, 1, 10)
    print("Матриця1: \n", matrix1)
    print("Матриця2: \n", matrix2)
    print("Різниця матриць: \n", matrixSub(matrix1, matrix2))

def task16():    
    print("\n-----\n#16-Створіть матрицю розміром 3x3, заповніть її випадковими дійсними числами\n \
    в діапазоні від 0 до 1 та знайдіть вектор-стовпчик, що містить суму елементів кожного рядка матриці\n")
    matrix = createMatrix(3, 3, 0, 1)
    print("Матриця: \n", np.round(matrix, 2))
    print("Вектор, що містить суму елементів кожного рядка матриці:\n", np.round(matrixSumNumsLine(matrix), 2))

def task17():   
    print("\n-----\n#17-Створіть матрицю розміром 3x4 з довільними цілими числами і створінь матрицю з квадратами цих чисел\n")
    matrix = createMatrix(3, 4, 1, 50)
    print("Матриця: \n", matrix)
    print("Матриця з квадратами чисел:\n", matrixDoubleNums(matrix))

def task18():   
    print("\n-----\n#18-Створіть вектор розміром 4, заповніть його випадковими цілими числами в діапазоні від 1 до 50 \n\
    та знайдіть вектор з квадратними коренями цих чисел\n")
    vect = createRandVector(4, 1, 50)
    print("Вектор початковий: ", vect)
    print("Вектор з квадратними коренями цих чисел: ", np.round(vectorSquardLine(vect), 2))


def main():
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()
    task8()
    task9()
    task10()
    task11()
    task12()
    task13()
    task14()
    task15()
    task16()
    task17()
    task18()

if __name__ == '__main__':
    main()