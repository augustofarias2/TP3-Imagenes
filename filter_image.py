import cv2
import numpy as np

def filtered(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_filtered = cv2.medianBlur(img_gray, 9)
    img_canny_CV2 = cv2.Canny(img_filtered, 30, 90)

    # Convertir el frame de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir el rango de colores rojos en HSV
    lower_red = np.array([0, 75, 20])
    upper_red = np.array([10, 255, 255])

    # Crear una máscara para los píxeles rojos
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Definir otro rango de colores rojos en HSV
    lower_red = np.array([160, 100, 20])
    upper_red = np.array([179, 255, 255])

    # Crear una máscara para los píxeles rojos en el rango 2
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combinar las máscaras para abarcar un rango más amplio de rojos
    mask = mask1 + mask2

    # Aplicar la máscara al frame original
    result = cv2.bitwise_and(frame, frame, mask=mask)

    img_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    img_filtered = cv2.medianBlur(img_gray, 7)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_img = cv2.morphologyEx(img_filtered, cv2.MORPH_OPEN, se)   # Apertura para remover elementos pequeños
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, se)  # Clausura para rellenar huecos.
    img_canny_CV2 = cv2.Canny(img_filtered, 30, 90)
    
    return binary_img