import cv2
import matplotlib.pyplot as plt
import numpy as np
from detect_dices import detectar_dado
from filter_image import filtered

def create_video(tirada):
    # --- Leer y grabar un video ------------------------------------------------
    cap = cv2.VideoCapture(f'tirada_{tirada}.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(f'Video-Output{tirada}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            
            img_filtered = filtered(frame)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_filtered)

            if num_labels > 12:
                frame=detectar_dado(frame, num_labels, labels, stats, centroids)

            # --- Muestro por pantalla ------------
            img_filtered_show = cv2.resize(img_filtered, dsize=(int(width/3), int(height/3)))
            frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
            # Mostrar el resultado
            cv2.imshow('Analisis de Dados', frame_show) # Usar img_filtered_show para ver el filtrado y frame_show para ver el resultado final
                    # ---------------------------------------------------------------
            out.write(frame)  # grabo frame --> IMPORTANTE: frame debe tener el mismo tamaño que se definio al crear out.
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


for i in range(1,5):
    create_video(i)

