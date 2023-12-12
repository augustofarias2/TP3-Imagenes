import cv2
import numpy as np

def circulars(contours): #Completar función para detectar agujeros circulares
    holes = 0
    return holes

def detectar_dado(frame, num_labels, labels, stats, centroids):
    labeled_shape = np.zeros_like(frame)
    RHO_TH = 0.8    # Factor de forma (rho)
    AREA_TH = 500   # Umbral de area
    aux = np.zeros_like(labels)
    labeled_image = cv2.merge([aux, aux, aux])
    # Clasifico en base al factor de forma
    for i in range(1, num_labels):

        # --- Remuevo celulas con area chica --------------------------------------
        if (stats[i, cv2.CC_STAT_AREA] < AREA_TH):
            continue

        # --- Selecciono el objeto actual -----------------------------------------
        obj = (labels == i).astype(np.uint8)

        # --- Calculo Rho ---------------------------------------------------------
        ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(ext_contours[0])
        perimeter = cv2.arcLength(ext_contours[0], True)
        rho = 4 * np.pi * area/(perimeter**2)
        sens = 0.35
        flag_cuadrado = (1-sens) * RHO_TH < rho < (1+sens) * RHO_TH
        
        # --- Clasifico -----------------------------------------------------------

        if flag_cuadrado:
            # --- Calculo cantidad de puntos ------------------------------------------
            all_contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            holes = circulars(all_countours) 
            if holes != 0:
                # --- Dibujo el bounding box ---------------------------------------------
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
                # --- Dibujo el label ----------------------------------------------------
                cv2.putText(frame, f"{holes}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame