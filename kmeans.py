import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# import random

class KMeans:
    def __init__(self, k=4, tol=0.001, max_iter=2000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # Inicializar los centroides
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            # Asignar cada punto al cluster más cercano
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(features)

            prev_centroids = dict(self.centroids)

            # Actualizar los centroides
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            # Verificar si los centroides han cambiado significativamente
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# funciones requeridas
def calculate_dominant_color(img):
    # Convertir la imagen de BGR a RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dividir la imagen en los canales R, G y B
    r, g, b = cv2.split(image)
    sRed = np.sum(r)
    sGreen = np.sum(g)
    sBlue = np.sum(b)

    # Calcular el color dominante
    colors = [sRed, sGreen, sBlue]
    dominant_color = colors.index(max(colors)) * 25

    return dominant_color


# Calcular la circularidad de la imagen
def calculate_circularity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity


# Calcular los momentos de Hu de la imagen
def calculate_hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments


def recImg(image_name, prediction):
    print('inicio')
    # Cargamos la imagen
    image = cv2.imread(image_name)

    # Definimos los colores para cada clase en formato BGR
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    # Convertimos la imagen a escala de grises y aplicamos un umbral
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

    # Encontramos los contornos en la imagen
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujamos un rectángulo alrededor de cada contorno
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + width, y + height), colors[prediction], 2)

            # Mostramos la imagen
            # pick = cv2.imread(image, 1)
            pick = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(pick)
            plt.show()
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


# Crear un conjunto de datos que incluya el color predominante, la circularidad y los momentos de Hu de cada imagen
# Direcciones de las imágenes
entrenamiento = "C:/Users/tguev/Documents/Fing/IA/Proyecto/Curso/redesNeuronales/DataSet/entrenamiento"
prueba = "C:/Users/tguev/Documents/Fing/IA/Proyecto/Curso/redesNeuronales/DataSet/Validacion"

listTrain = os.listdir(entrenamiento)
listTest = os.listdir(prueba)

# Parámetros
ancho, alto = 200, 200

data = []
# Cargar imágenes de entrenamiento
for nameDir in listTrain:
    nombre = entrenamiento + "/" + nameDir  # Leemos las fotos
    for nameFile in os.listdir(nombre):  # asignamos etiquetas
        img = cv2.imread(os.path.join(nombre, nameFile))
        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        dominant_color = calculate_dominant_color(img)
        circularity = calculate_circularity(img)
        hu_moments = calculate_hu_moments(img)
        data.append(np.hstack([dominant_color, circularity, np.ravel(hu_moments)]))

# Crear una instancia de KMeans
kmeans = KMeans(k=4)

# Ajustar los datos
kmeans.fit(data)

# Visualizar los resultados

# hacer una prueba de predicción
# imgN = cv2.imread("C:/Users/tguev/Documents/Fing/IA/Proyecto/Curso/redesNeuronales/DataSet/Validacion/banana/20240106_171414.jpg")
# "C:\Users\tguev\Documents\Fing\IA\Proyecto\Curso\redesNeuronales\DataSet\Validacion\naranja\naranja5.jpg"
# "C:/Users/tguev/Documents\Fing\IA\Proyecto\Curso/redesNeuronales\DataSet\Validacion\manzana\img_791.jpeg"
# C:\Users\tguev\Downloads\Quick Share\20240108_173308.jpg
imgN = cv2.imread("C:\\Users\\tguev\Downloads\Quick Share\\20240212_112037.jpg")
imgDir = "C:\\Users\\tguev\Downloads\Quick Share\\20240212_112037.jpg"

imgN = cv2.resize(imgN, (ancho, alto), interpolation=cv2.INTER_CUBIC)
dominant_color = calculate_dominant_color(imgN)
circularity = calculate_circularity(imgN)
hu_moments = calculate_hu_moments(imgN)
data = np.hstack([dominant_color, circularity, np.ravel(hu_moments)])
prediction = kmeans.predict(data)
recImg(imgDir, prediction)
print(prediction)

# manzana = 0 -- 1
# naranja = 3 -- 0
# banana = 2
# pera = 1
