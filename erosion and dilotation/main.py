import cv2
import numpy as np
import os


def erode(image, kernel_size=10):
    pad_size = kernel_size // 2
    image_padded = np.pad(image, pad_width=pad_size, mode="constant", constant_values=0)
    eroded_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = image_padded[i: i + kernel_size, j: j + kernel_size]
            eroded_image[i, j] = 255 if np.all(region == 255) else 0

    return eroded_image


def dilate(image, kernel_size=10):
    pad_size = kernel_size // 2
    image_padded = np.pad(image, pad_width=pad_size, mode="constant", constant_values=0)
    dilated_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = image_padded[i: i + kernel_size, j: j + kernel_size]
            dilated_image[i, j] = 255 if np.any(region == 255) else 0

    return dilated_image


def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_name = os.path.basename(image_path).split(".")[0]

    if img is None:
        raise ValueError("Нет изображения в папке с таким названием")

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    operation_count = 1
    while True:
        print("Выберите операцию:")
        print("1. Эрозия")
        print("2. Дилатация")
        print("3. Выход")

        choice = input("Введите номер операции: ")

        if choice == "1" or choice == "2":
            kernel_size = int(input("Введите размер ядра: "))
            if choice == "1":
                binary = erode(binary, kernel_size)
                operation_type = "erosion"
            elif choice == "2":
                binary = dilate(binary, kernel_size)
                operation_type = "dilation"

            output_dir = f"imgs/{operation_type}s"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{img_name}_{operation_count}_{kernel_size}.png"
            cv2.imwrite(output_path, binary)
            print(f"Изображение сохранено: {output_path}")
            operation_count += 1

        elif choice == "3":
            break
        else:
            print("Неправильный выбор. Попробуйте снова.")


if __name__ == "__main__":
    image_path = input("Введите название изображения (например, chocopie.png): ")
    process_image(f"imgs/{image_path}")
