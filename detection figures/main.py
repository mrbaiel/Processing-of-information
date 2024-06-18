from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def apply_binarization(image):
    """
    Применяет пороговые значения к изображению и возвращает бинарное изображение.

    Параметры:
    image (numpy.ndarray): Исходное изображение.

    Возвращает:
    numpy.ndarray: Бинарное изображение, где пиксели меньше 127 равны 0,
    а пиксели больше или равные 127 равны 255.
    """
    binary_image = np.zeros_like(image)
    binary_image[np.where(image < 127)] = 0
    binary_image[np.where(image >= 127)] = 255
    return binary_image


def mark_image(labeled_image, properties):
    """
    Рисует метки и свойства каждой метки на изображении.

    Параметры:
    labeled_image (numpy.ndarray): Массив с метками компонентов.
    properties (list[dict]): Список словарей, где каждый словарь содержит информацию о каждой метке.

    Возвращает:
    PIL.Image.Image: Цветное изображение с метками и свойствами.
    """

    color_image = Image.fromarray(
        np.uint8(labeled_image * 255 // np.max(labeled_image))
    )
    color_image = color_image.convert("RGB")
    draw = ImageDraw.Draw(color_image)

    try:

        font = ImageFont.truetype("arial.ttf", size=12)
    except IOError:

        font = ImageFont.load_default()

    for prop in properties:
        label = prop["label"]
        center_of_mass = prop["center_of_mass"]

        cm_x, cm_y = int(center_of_mass[0]), int(center_of_mass[1])
        draw.ellipse((cm_x - 2, cm_y - 2, cm_x + 2, cm_y + 2), fill="red")

        mask = labeled_image == label
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]

                    if any(
                        ny < 0
                        or ny >= mask.shape[0]
                        or nx < 0
                        or nx >= mask.shape[1]
                        or not mask[ny, nx]
                        for ny, nx in neighbors
                    ):
                        draw.point((x, y), "green")

                    if (
                        sum(
                            1
                            for ny, nx in neighbors
                            if 0 <= ny < mask.shape[0]
                            and 0 <= nx < mask.shape[1]
                            and mask[ny, nx]
                        )
                        < 4
                    ):
                        draw.point((x, y), "blue")

        text_position = (cm_x + 5, cm_y - 5)
        draw.text(text_position, str(label), font=font, fill="yellow")

    return color_image


def bfs_component_labeling(image):
    """
    Применяет алгоритм обхода в ширину для маркировки компонентов на бинарном изображении.

    Параметры:
    image (numpy.ndarray): Бинарное изображение, где 0 - это фон, 1 - объект.

    Возвращает:
    numpy.ndarray: Массив с метками компонентов.
    """
    rows, cols = image.shape
    labeled = np.zeros_like(image)
    label = 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 1 and labeled[i, j] == 0:
                label += 1
                queue = deque([(i, j)])
                labeled[i, j] = label
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (
                                0 <= nx < rows
                                and 0 <= ny < cols
                                and image[nx, ny] == 1
                                and labeled[nx, ny] == 0
                        ):
                            queue.append((nx, ny))
                            labeled[nx, ny] = label
    return labeled


def calculate_properties(labeled_image):
    """
    Вычисляет свойства каждой метки на изображении.

    Параметры:
    labeled_image (numpy.ndarray): Массив с метками компонентов.

    Возвращает:
    list[dict]: Список словарей, где каждый словарь содержит информацию о каждой метке.
    """
    properties = []
    labels = np.unique(labeled_image)
    labels = labels[labels != 0]

    for label in labels:
        mask = labeled_image == label
        coords = np.column_stack(np.where(mask))
        area = coords.shape[0]
        center_of_mass = coords.mean(axis=0)[::-1]  # Invert to (x, y)

        # Вычисление размеров (ширина и высота)
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        perimeter = np.sum(mask[1:-1, 1:-1] & (~mask[:-2, 1:-1] | ~mask[2:, 1:-1] | ~mask[1:-1, :-2] | ~mask[1:-1, 2:]))

        circ_coefficient = (perimeter ** 2) / area

        properties.append({
            "label": label,
            "area": area,
            "center_of_mass": center_of_mass,
            "perimeter": perimeter,
            "circ_coefficient": circ_coefficient,
            "width": width,
            "height": height
        })

    return properties


def calculate_angle(p1, p2, p3):
    """
    Вычисляет угол между тремя точками p1, p2, p3.

    Параметры:
    p1 (tuple): Координаты первой точки.
    p2 (tuple): Координаты второй точки (вершина угла).
    p3 (tuple): Координаты третьей точки.

    Возвращает:
    float: Угол в градусах.
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitudes = np.sqrt(v1[0] ** 2 + v1[1] ** 2) * np.sqrt(v2[0] ** 2 + v2[1] ** 2)
    angle = np.arccos(dot_product / magnitudes)
    return np.degrees(angle)


def main():
    image_name = input("Введите название файла изображения вместе с расширением (image.jpg, картинка.png): ")
    image_path = f"imgs/{image_name}"
    image = Image.open(image_path).convert("L")
    print(f"Размер изображения: {image.size}")
    image = np.array(image)

    binary_img = apply_binarization(image)
    print(f"Размер бинарного изображения: {binary_img.shape}")

    binary_img_save_path = f"imgs/binary_imgs/{image_name}_binary.jpg"
    Image.fromarray(binary_img).save(binary_img_save_path)

    binary_img = 255 - binary_img
    binary_img[binary_img == 255] = 1
    np.savetxt(f"txt/{image_name}_binary.txt", binary_img, fmt="%d")

    labeled_image = bfs_component_labeling(binary_img)
    print(f"Размер маркированного изображения: {labeled_image.shape}")

    np.savetxt(f"txt/{image_name}_labeled.txt", labeled_image, fmt="%d")
    properties = calculate_properties(labeled_image)

    print("Количество фигур:", len(properties))
    for prop in properties:
        print(f"Фигура {prop['label']} - Площадь: {prop['area']}, Координаты центра масс: {prop['center_of_mass']}, "
              f"Периметр: {prop['perimeter']}, Коэффициент круглости: {prop['circ_coefficient']}, "
              f"Ширина: {prop['width']}, Высота: {prop['height']}")

    marked_image = mark_image(labeled_image, properties)
    print(f"Размер изображения с метками: {marked_image.size}")
    marked_image_path = f"imgs/marked_imgs/{image_name}_marked.jpg"
    marked_image.save(marked_image_path)

if __name__ == "__main__":
    main()



# from collections import deque
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
#
#
# def apply_binarization(image):
#     """
#     Применяет пороговые значения к изображению и возвращает бинарное изображение.
#
#     Параметры:
#     image (numpy.ndarray): Исходное изображение.
#
#     Возвращает:
#     numpy.ndarray: Бинарное изображение, где пиксели меньше 127 равны 0,
#     а пиксели больше или равные 127 равны 255.
#     """
#     return np.where(image < 127, 0, 255)
#
#
# def mark_image(labeled_image, properties):
#     """
#     Рисует метки и свойства каждой метки на изображении.
#
#     Параметры:
#     labeled_image (numpy.ndarray): Массив с метками компонентов.
#     properties (list[dict]): Список словарей, где каждый словарь содержит информацию о каждой метке.
#
#     Возвращает:
#     PIL.Image.Image: Цветное изображение с метками и свойствами.
#     """
#     color_image = Image.fromarray(np.uint8(labeled_image * 255 // np.max(labeled_image)))
#     color_image = color_image.convert("RGB")
#     draw = ImageDraw.Draw(color_image)
#
#     try:
#         font = ImageFont.truetype("arial.ttf", size=12)
#     except IOError:
#         font = ImageFont.load_default()
#
#     for prop in properties:
#         label = prop["label"]
#         cm_x, cm_y = map(int, prop["center_of_mass"])
#         draw.ellipse((cm_x - 2, cm_y - 2, cm_x + 2, cm_y + 2), fill="red")
#
#         mask = labeled_image == label
#         border_mask = np.zeros_like(mask)
#         border_mask[1:-1, 1:-1] = mask[1:-1, 1:-1] & (~mask[:-2, 1:-1] | ~mask[2:, 1:-1] | ~mask[1:-1, :-2] | ~mask[1:-1, 2:])
#         border_pixels = np.argwhere(border_mask)
#
#         for y, x in border_pixels:
#             if mask[y, x]:
#                 neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
#                 if any(
#                     ny < 0 or ny >= mask.shape[0] or nx < 0 or nx >= mask.shape[1] or not mask[ny, nx]
#                     for ny, nx in neighbors
#                 ):
#                     draw.point((x, y), "green")
#
#                 if sum(
#                     1 for ny, nx in neighbors
#                     if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]
#                 ) < 4:
#                     draw.point((x, y), "blue")
#
#         text_position = (cm_x + 5, cm_y - 5)
#         draw.text(text_position, str(label), font=font, fill="yellow")
#
#     return color_image
#
#
# def bfs_component_labeling(image):
#     """
#     Применяет алгоритм обхода в ширину для маркировки компонентов на бинарном изображении.
#
#     Параметры:
#     image (numpy.ndarray): Бинарное изображение, где 0 - это фон, 1 - объект.
#
#     Возвращает:
#     numpy.ndarray: Массив с метками компонентов.
#     """
#     rows, cols = image.shape
#     labeled = np.zeros_like(image, dtype=int)
#     label = 1
#
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#
#     for i in range(rows):
#         for j in range(cols):
#             if image[i, j] == 1 and labeled[i, j] == 0:
#                 queue = deque([(i, j)])
#                 labeled[i, j] = label
#                 while queue:
#                     x, y = queue.popleft()
#                     for dx, dy in directions:
#                         nx, ny = x + dx, y + dy
#                         if 0 <= nx < rows and 0 <= ny < cols and image[nx, ny] == 1 and labeled[nx, ny] == 0:
#                             queue.append((nx, ny))
#                             labeled[nx, ny] = label
#                 label += 1
#     return labeled
#
#
# def calculate_properties(labeled_image):
#     """
#     Вычисляет свойства каждой метки на изображении.
#
#     Параметры:
#     labeled_image (numpy.ndarray): Массив с метками компонентов.
#
#     Возвращает:
#     list[dict]: Список словарей, где каждый словарь содержит информацию о каждой метке.
#     """
#     properties = []
#     labels = np.unique(labeled_image)
#     labels = labels[labels != 0]
#
#     for label in labels:
#         mask = labeled_image == label
#         coords = np.column_stack(np.where(mask))
#         area = coords.shape[0]
#         center_of_mass = coords.mean(axis=0)[::-1]  # Invert to (x, y)
#
#         perimeter = np.sum(mask[1:-1, 1:-1] & (~mask[:-2, 1:-1] | ~mask[2:, 1:-1] | ~mask[1:-1, :-2] | ~mask[1:-1, 2:]))
#
#         circ_coefficient = (perimeter**2) / area
#
#         properties.append({
#             "label": label,
#             "area": area,
#             "center_of_mass": center_of_mass,
#             "perimeter": perimeter,
#             "circ_coefficient": circ_coefficient,
#         })
#
#     return properties
#
#
# def main():
#     image_name = input("Введите название файла изображения вместе с расширением (image.jpg, картинка.png): ")
#     image_path = f"imgs/{image_name}"
#     image = Image.open(image_path).convert("L")
#     image = np.array(image)
#     binary_img = apply_binarization(image)
#     binary_img_save_path = f"imgs/binary_imgs/{image_name}_binary.jpg"
#     Image.fromarray(binary_img.astype(np.uint8)).save(binary_img_save_path)
#     binary_img = 255 - binary_img
#     binary_img[binary_img == 255] = 1
#     np.savetxt(f"txt/{image_name}_binary.txt", binary_img, fmt="%d")
#
#     labeled_image = bfs_component_labeling(binary_img)
#
#     np.savetxt(f"txt/{image_name}_labeled.txt", labeled_image, fmt="%d")
#     properties = calculate_properties(labeled_image)
#
#     print("Количество фигур:", len(properties))
#     for prop in properties:
#         print(f"Фигура {prop['label']} - Площадь: {prop['area']}, Координаты центра масс: {prop['center_of_mass']}, Периметр: {prop['perimeter']}, Коэффициент круглости: {prop['circ_coefficient']}")
#
#     marked_image = mark_image(labeled_image, properties)
#     marked_image_path = f"imgs/marked_imgs/{image_name}_marked.jpg"
#     marked_image.save(marked_image_path)
#
#
# if __name__ == "__main__":
#     main()
