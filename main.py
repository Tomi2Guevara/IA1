from kmeans2 import KMeans
from kmeans2 import foto
from knn import KNN
import cv2
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np

def audio(link):
    y, sr = knn.preprocess_audio(link)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    dataTest = np.concatenate((mfccs*100, chroma*0.5, contrast*0.2, tonnetz*0.2), axis=0)
    return dataTest

def photo(link):
    img = cv2.imread(link)
    img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    h, s = km.calculate_dominant_color(img)
    imgHu, imgCir = km.preprocess_image(img)
    circularity = km.calculate_circularity(imgCir)
    hu_moments = km.calculate_hu_moments(imgHu)
    return h, s, circularity, hu_moments

# Crear un conjunto de datos que incluya el color predominante, la circularidad y los momentos de Hu de cada imagen
# Direcciones de las imágenes
entrenamiento = r"C:\Users\tguev\Documents\Fing\IA\Curso\Fotos"

listTrain = os.listdir(entrenamiento)

# Parámetros
ancho, alto = 200, 200
km = KMeans(4)

data = []
centros = []
# Cargar imágenes de entrenamiento
for nameDir in listTrain:
    nombre = entrenamiento + "/" + nameDir  # Leemos las fotos
    for nameFile in os.listdir(nombre):  # asignamos etiquetas
        file_path = os.path.join(nombre, nameFile)
        h, s, circularity, hu_moments = photo(file_path)
        # instanciar objeto del tipo foto
        newFoto = foto(h, s, circularity, hu_moments, nameFile)
        data.append(newFoto)
    centros.append(newFoto.car)

# Ajustar los datos
centroides = km.fit(data, centros)


#----------------------- AUDIO -----------------------

# Cargar y preprocesar los audios
entrenamiento = r"C:\Users\tguev\Documents\Fing\IA\Curso\audios"

listTrain = os.listdir(entrenamiento)

knn = KNN(5)

xTrain = []
yTrain = []
# Cargar audios de entrenamiento
for nameDir in listTrain:
    nombre = entrenamiento + "//" + nameDir  # Leemos las fotos
    for nameFile in os.listdir(nombre):  # asignamos etiquetas
        file_path = os.path.join(nombre, nameFile)
        data = audio(file_path)
        xTrain.append(data)
        yTrain.append(knn.label_for_filename(nombre))

# Crear una instancia de KNN
knn = KNN(5)

# Ajustar los datos (asumiendo que 'y' es tu vector de etiquetas)
knn.fit(xTrain, yTrain)

#----------------------- USO -----------------------

#clasificar las fotos
fotos = r"C:\Users\tguev\Documents\Fing\IA\Curso\test\Fotos"
listTrain = os.listdir(fotos)
fotoPred = []
paths = []
for nameDir in listTrain:
    nombre = fotos + "/" + nameDir  # Leemos las fotos
    file_path = os.path.join(nombre)
    h, s, circularity, hu_moments = photo(file_path)
    newFoto0 = foto(h, s, circularity, hu_moments, nameDir)
    fotoPred.append(km.predict(newFoto0))
    paths.append(file_path)



#cargamos el audio a identificar
test = r"C:\Users\tguev\Documents\Fing\IA\Curso\test\Audios"
testList = os.listdir(test)
test = test + "\\" + testList[0]
dataTest = audio(test)
comando = knn.predict(dataTest)

for i in fotoPred:
    if i == comando:
        print("El audio es: ", comando)
        #plotear la fruta, usnado matplotlib
        fig, ax = plt.subplots()
        img = cv2.imread(paths[fotoPred.index(i)])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        plt.show()
        break


