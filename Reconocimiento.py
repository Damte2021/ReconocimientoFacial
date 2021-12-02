import cv2
import os
import mediapipe as mp

#-------------importar los nombres de las carpetas------------------
direccion = 'C:/Users/sofia/Desktop/rf/proyectoRF2/Fotos'
etiquetas = os.listdir(direccion)
print("Nombre: ", etiquetas)

#--------------llamar al modelo entrenado--------------
modelo = cv2.face.LBPHFaceRecognizer_create()

#_____________leer el modelo__________________
modelo.read("modeloEntrenado.xml")

#-----------------------declaracion deldetector--------------------------
detector = mp.solutions.face_detection
dibujo = mp.solutions.drawing_utils


#----------------------video captura de rostros--------------------------
cap = cv2.VideoCapture(0)
#inicializamos el contador
cont = 0

#-----------------------inicilaizar parametos de deteccion---------------
with detector.FaceDetection(min_detection_confidence= 0.75) as rostros:
    #inicializo while true
    while True:
        #lectura de videocaptura
        ret, frame = cap.read()
        copia = frame.copy()

        # elimiar error de movimiento
        frame = cv2.flip(copia, 1)

        # correccion de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copia2 = rgb.copy()

        # deteccion de rostros
        resultado = rostros.process(copia2)

        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame, rostro)
                #print(rostro)

                #extraer el ancho y el alto de la ventana
                al, an, _ = frame.shape

                #extraer x inicial y y inicial
                xi = rostro.location_data.relative_bounding_box.xmin
                yi = rostro.location_data.relative_bounding_box.ymin

                #extraer ancho y alto
                ancho = rostro.location_data.relative_bounding_box.width
                alto = rostro.location_data.relative_bounding_box.height

                #conversion a pixeles
                xi = int(xi * an)
                yi = int(yi * al)
                ancho = int(ancho * an)
                alto = int(alto * al)

                #encontrar yfinal y xfinal
                xf = xi + ancho
                yf = yi + alto

                # extraer el punto central del rostro
                cx = (xi + (xi + xf)) // 2
                cy = (yi + (yi + yf)) // 2

                #extraccion de pixeles
                cara = copia2[yi:yf, xi:xf]

                #redimencionar las fotos
                cara = cv2.resize(cara, (300,300), interpolation=cv2.INTER_CUBIC)
                cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)

                #realizar la prediccion
                prediccion = modelo.predict(cara)
                print(prediccion)

                #mostrar los resultados en pantalla
                if prediccion[1] < 38:
                    cv2.putText(frame, '{}'.format(etiquetas[0]), (xi, yi -5), 1, 1.3, (0,0,255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi,yi), (xf, yf), (0,0,255), 2)
                #else:
                elif prediccion[1] > 39:
                    cv2.putText(frame, 'Desconocido', (xi, yi), 1, 1.3, (255,0,0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi,yi), (xf, yf), (255,0,0), 2)


                #else:
                    #cv2.putText(frame, 'Desconocido', (xi, yi - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                    #cv2.rectangle(frame, (xi, yi), (xf, yf), (0, 0, 255), 2)




        #mostramos los fotogramas
        cv2.imshow("Reconocimiento facial", frame)
        #leo la tecla escape
        t = cv2.waitKey(1)
        if t == 27:
            break

cap.release()
cv2.destroyAllWindows()
