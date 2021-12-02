import cv2
import numpy as np
import os

#-----------------------importar fotos tomasdas antoeriormente.--------------
direccion = 'C:/Users/sofia/Desktop/rf/proyectoRF2/Fotos'
lista = os.listdir(direccion)

etiquetas = []
rostros = []
cont = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir

    for filaName in os.listdir(nombre):
        etiquetas.append(cont) #asignamos las etiquetas
        rostros.append(cv2.imread(nombre + '/' + filaName,0))


    cont = cont + 1

#----------------------------creamos el modelo--------------------------
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

#-------------------------------entrenamos el modelo--------------------
reconocimiento.train(rostros, np.array(etiquetas))

#----------------------guardamos el modelo-----------------------------
reconocimiento.write("modeloEntrenado.xml")
print("modelo creado")