import cv2
import mediapipe as mp
import os

#-------------------------creacion de carpeta donde se van a guardar las fotos------------
nombre = "Sofia"
direccion = 'C:/Users/sofia/Desktop/rf/proyectoRF2/Fotos'
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print("Carpeta creada")
    os.makedirs(carpeta)

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

        # elimiar error de movimiento
        frame = cv2.flip(frame, 1)

        # correccion de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # deteccion de rostros
        resultado = rostros.process(rgb)

        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame, rostro)
                #print(rostro)

                #extraer el ancho y el alto de la ventana
                an, al, _ = frame.shape

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

                #extraccion de pixeles
                cara = frame[yi:yf, xi:xf]

                #redimencionar las fotos
                cara = cv2.resize(cara, (150, 200), interpolation=cv2.INTER_CUBIC)

                #almacenar mis imagenes
                cv2.imwrite(carpeta + "/rostro_{}.jpg".format(cont), frame)
                cont = cont + 1



        #mostramos los fotogramas
        cv2.imshow("Reconocimiento facial Sofia", frame)
        #leo la tecla escape
        t = cv2.waitKey(1)
        if t == 27 or cont > 300:
            break

cap.release()
cv2.destroyAllWindows()


