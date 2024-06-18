import numpy as np


def get_letter(matrix):
    letters = {
        "A": np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
            ]
        ),
        "Б": np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        ),
        "В": np.array(
            [
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        ),
        "Г": np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ]
        ),
        "Е": np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        ),
    }
    min_diff = 1000
    letter = ""
    for key, value in letters.items():
        diff = np.sum(value != matrix)
        if diff < min_diff:
            min_diff = diff
            letter = key
    return letter


def main():
    # matrix = np.array([[1, 1, 1, 1, 0],
    #                     [1, 0, 0, 0, 1],
    #                     [1, 1, 1, 1, 0],
    #                     [1, 0, 0, 0, 1],
    #                     [1, 1, 1, 1, 0]])
    matrix = np.zeros((5, 5))
    print("Введите матрицу 5x5 (формат '1 0 1 0 1'):")
    for i in range(5):
        print(f"Строка №{i + 1}")
        matrix[i] = list(map(int, input().split()))
    print(get_letter(matrix))


if __name__ == "__main__":
    main()
