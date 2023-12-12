import cv2
import matplotlib.pyplot as plt
import numpy as np
from detect_dices import detectar_dado
from filter_image import filtered

def euclidean_distance(list1, list2):
    # Asegurarse de que las listas tengan la misma longitud
    list1 = sorted(list1, key=len)
    list2 = sorted(list2, key=len)
    min_len = min(len(list1), len(list2))
    
    # Calcular la distancia Euclidiana
    distance = np.linalg.norm(np.array(list1[:min_len]) - np.array(list2[:min_len]))

    return distance

def manhattan_distance(list1, list2):
    list1 = sorted(list1, key=len)
    list2 = sorted(list2, key=len)
    return sum(sum(abs(x - y) for x, y in zip(tuple1, tuple2)) for tuple1, tuple2 in zip(list1, list2))

def create_video(tirada):
    # --- Leer y grabar un video ------------------------------------------------
    cap = cv2.VideoCapture(f'tirada_{tirada}.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(f'Video-Output{tirada}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    centroids1=[[0,0]]
    fr=0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            
            img_filtered = filtered(frame)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_filtered)
            
            distancia = manhattan_distance(centroids1, centroids)
            #cv2.putText(frame, f"Frame {fr}:{distancia}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #print("Frame: ", fr, "Distancia: ", distancia)
            # if distancia < 1200:
            frame=detectar_dado(frame, num_labels, labels, stats, centroids)
            centroids1 = centroids

            # --- Muestro por pantalla ------------
            img_filtered_show = cv2.resize(img_filtered, dsize=(int(width/3), int(height/3)))
            frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
            # Mostrar el resultado
            cv2.imshow(f'Analisis de Dados Video {tirada}', img_filtered_show) # Usar img_filtered_show para ver el filtrado y frame_show para ver el resultado final
                    # ---------------------------------------------------------------
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        fr+=1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# for i in range(1, 5):
#     create_video(i)

create_video(6)