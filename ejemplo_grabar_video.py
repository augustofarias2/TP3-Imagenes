import cv2
import matplotlib.pyplot as plt
import numpy as np

def filtered(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_filtered = cv2.medianBlur(img_gray, 9)
    img_canny_CV2 = cv2.Canny(img_filtered, 30, 90)

    # Convertir el frame de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir el rango de colores rojos en HSV
    lower_red = np.array([0, 100, 20])
    upper_red = np.array([8, 255, 255])

    # Crear una máscara para los píxeles rojos
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Definir otro rango de colores rojos en HSV
    lower_red = np.array([175, 100, 20])
    upper_red = np.array([179, 255, 255])

    # Crear una máscara para los píxeles rojos en el rango 2
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combinar las máscaras para abarcar un rango más amplio de rojos
    mask = mask1 + mask2

    # Aplicar la máscara al frame original
    result = cv2.bitwise_and(frame, frame, mask=mask)

    img_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    img_filtered = cv2.medianBlur(img_gray, 7)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_img = cv2.morphologyEx(img_filtered, cv2.MORPH_OPEN, se)   # Apertura para remover elementos pequeños
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, se)  # Clausura para rellenar huecos.
    img_canny_CV2 = cv2.Canny(img_filtered, 30, 90)
    
    return binary_img

def detectar_dado(img, num_labels, labels, stats, centroids):
    labeled_shapes = np.zeros_like(img)
    RHO_TH = 0.8    # Factor de forma (rho)
    AREA_TH = 500   # Umbral de area
    aux = np.zeros_like(labels)
    labeled_image = cv2.merge([aux, aux, aux])
    # Clasifico en base al factor de forma
    holes=[]
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
        flag_circular = rho > RHO_TH
        
        # --- Clasifico -----------------------------------------------------------

        if flag_circular:
            pass
        else:
            # --- Calculo cantidad de puntos ------------------------------------------
            all_contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hole = len(all_contours) - 1
            holes.append(hole)
        
    cv2.putText(img_filtered,f"[{', '.join(map(str, holes))}]", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)    

# --- Leer y grabar un video ------------------------------------------------
cap = cv2.VideoCapture('tirada_2.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('Video-Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
flag=0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        
        img_filtered = filtered(frame)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_filtered)
        
        print(num_labels)

        if num_labels > 12:
            detectar_dado(img_filtered, num_labels, labels, stats, centroids)
            flag+=1

        if flag == 12:
            cv2.putText(img_filtered,"6", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            final_labels = labels
            final_stats = stats
            final_num_labels = num_labels

        # print(num_labels)

        out.write(img_filtered)
        # --- Procesamiento ---------------------------------------------
        # cv2.rectangle(frame, (100,100), (200,200), (0,0,255), 2)

        # --- Muestro por pantalla ------------
        frame_show = cv2.resize(img_filtered, dsize=(int(width/3), int(height/3)))
        # Mostrar el resultado
        cv2.imshow('Segmentación Roja', frame_show)
                # ---------------------------------------------------------------
        out.write(frame)  # grabo frame --> IMPORTANTE: frame debe tener el mismo tamaño que se definio al crear out.
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()



# --- Defino parametros para la clasificación -------------------------------------------
# RHO_TH = 0.8	# Factor de forma (rho), si es circulo el valor es mayor a 0.8
AREA_TH = 5000   # Umbral de area para descartar los labels que no sean figuras
aux = np.zeros_like(final_labels)
labeled_image = cv2.merge([aux, aux, aux])

# --- Clasificación ---------------------------------------------------------------------
# Clasifico en base al factor de forma
for i in range(1, final_num_labels):

    # # --- Remuevo celulas con area chica --------------------------------------
    # if (stats[i, cv2.CC_STAT_AREA] < AREA_TH):
    #     continue

    # --- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Calculo Rho ---------------------------------------------------------
    # ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # area = cv2.contourArea(ext_contours[0])
    # perimeter = cv2.arcLength(ext_contours[0], True)
    # rho = 4 * np.pi * area/(perimeter**2)
    # flag_circular = rho > RHO_TH

    # --- Clasifico -----------------------------------------------------------
    if final_stats[i, cv2.CC_STAT_AREA] < AREA_TH:
        labeled_image[obj == 1, 2] = 255
    else:
        labeled_image[obj == 1, 1] = 255
    
plt.imshow(labeled_image, cmap="gray"); plt.show(block=False)
