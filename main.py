from kmeans2 import KMeans
from kmeans2 import foto
from knn import KNN
import cv2
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.decomposition import PCA


def audio(link):
    y, sr = knn.preprocess_audio(link)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    zcr = librosa.feature.zero_crossing_rate(y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)

    dataTest = np.concatenate((mfccs, zcr), axis=0)
    return dataTest

def photo(link):
    img = cv2.imread(link)
    img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    h, s = km.calculate_dominant_color(img)
    imgHu, imgCir = km.preprocess_image(img)
    circularity = km.calculate_circularity(imgCir)
    hu_moments = km.calculate_hu_moments(imgHu)
    return h, s, circularity*50, hu_moments

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

#clasificar las fotos de entrenamiento
for i in data:
    print(i.id)
    print(km.predict(i))

#----------------------- GRAFICOS -----------------------
dataPlot = []
for i in data:
    dataPlot.append(i.car)

# Crea una instancia de PCA
pca = PCA(n_components=3)

# Crea un mapa de colores
colores = ['b', 'g', 'r', 'c']

# Ajusta y transforma los datos
data_pca = pca.fit_transform(dataPlot)

# Ajusta y transforma los centroides
centroides_pca = pca.transform(centroides)

# Crea una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Traza los datos
for i in range(len(data_pca)):
    ax.scatter(data_pca[i, 0], data_pca[i, 1], data_pca[i, 2], color=colores[km.predict(data[i])])

# Traza los centroides
for i in range(len(centroides_pca)):
    ax.scatter(centroides_pca[i, 0], centroides_pca[i, 1], centroides_pca[i, 2], color=colores[i], s=100, marker='x')

# Muestra el gráfico
plt.show()
x = 0
if x == 0:
    #terminar el programa
    exit()

#----------------------- AUDIO -----------------------

# Cargar y preprocesar los audios
entrenamiento = r"C:\Users\tguev\Documents\Fing\IA\Curso\audios"

listTrain = os.listdir(entrenamiento)

knn = KNN(3)

xTrain = []
yTrain = []
# Cargar audios de entrenamiento
for nameDir in listTrain:
    nombre = entrenamiento + "//" + nameDir  # Leemos las fotos
    for nameFile in os.listdir(nombre):  # asignamos etiquetas
        file_path = os.path.join(nombre, nameFile)
        sound = audio(file_path)
        xTrain.append(sound)
        yTrain.append(knn.label_for_filename(nombre))

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
    print("la foto es", i)
    if i == comando:
        print("El audio es: ", comando)
        #plotear la fruta, usnado matplotlib
        fig, ax = plt.subplots()
        img = cv2.imread(paths[fotoPred.index(i)])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        plt.show()



