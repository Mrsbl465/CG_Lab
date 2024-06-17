import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import matplotlib.pyplot as plt

def calculate_score(arr, angle):
    """
    Calcula un puntaje basado en el histograma de la imagen rotada.

    Args:
    - arr: Imagen binaria (thresh) sobre la cual se calcula el puntaje.
    - angle: Ángulo de rotación para aplicar a la imagen.

    Returns:
    - histogram: Histograma de la imagen rotada.
    - score: Puntaje calculado basado en el cambio del histograma.
    """
    data = inter.rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1, dtype=float)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
    return histogram, score


def rotate(image, delta=1, limit=6):
    """
    Corrige la inclinación de una imagen utilizando transformación de rotación.

    Args:
    - image: Imagen de entrada en formato OpenCV (numpy array).
    - delta: Incremento entre ángulos de rotación a probar.
    - limit: Límite de ángulos de rotación a probar (desde -limit hasta limit).

    Returns:
    - best_angle: Ángulo de rotación encontrado como el mejor para corregir la inclinación.
    - corrected: Imagen corregida con la inclinación corregida.
    """

    # Convierte la imagen a escala de grises y luego a imagen binaria invertida
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)  # Lista de ángulos 
    for angle in angles:
        histogram, score = calculate_score(thresh, angle)
        scores.append(score)

    # Encuentra el ángulo que tiene el puntaje máximo
    best_angle = angles[scores.index(max(scores))]

    # Aplica la rotación corregida a la imagen original
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)

    # Calcula las dimensiones de la imagen corregida para evitar recortes
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Ajusta la matriz de transformación para mover la imagen al centro
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Aplica la transformación 
    corrected = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    (h_corr, w_corr) = corrected.shape[:2]
    if h_corr > w_corr:
        rotated_90 = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)
        rotated_270 = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)

        hist_90, score_90 = calculate_score(rotated_90, 0)
        hist_270, score_270 = calculate_score(rotated_270, 0)

        if score_90 > score_270:
            corrected = rotated_90
            best_angle = 90
        else:
            corrected = rotated_270
            best_angle = 270

    filename = 'corrected_final.jpg'  
    cv2.imwrite(filename, corrected)

    return best_angle, corrected


image = cv2.imread('4.jpg')

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Base')  
plt.show()

#angle, corrected = rotate(image)

# 3 y 4
angle, corrected = rotate(image, 1 ,180 )

plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
plt.title('Corregida') 
plt.show()

print('Ángulo de inclinación corregido:', angle)

