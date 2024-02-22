import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# psr

class foto:
    def __init__(self, h, s, circularity, hu, id=None):
        self.id = id
        self.car = [h, s, circularity]
        for i in hu:
            self.car.append(i)

    def resta(self, other):
        return np.linalg.norm(np.array(self.car) - np.array(other))


class KMeans:
    def __init__(self, k=4, tol=0.001, max_iter=2000, centroids=None):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = centroids

    def fit(self, data, centroids):
        classifications = []
        cont = 0
        status = True
        while status:
            for j in range(self.k):
                classifications.append([])
            for i in data:
                # comparamos los atributos de data[i] con los atributos da cada centroide
                for centroid in centroids:
                    distances = i.resta(centroid)

                classification = np.argmin(distances)
                classifications[classification].append(i.car)
            # creamos los nuevos centroides
            centroidsNew = []
            for dot in range(self.k):
                if len(classifications[dot]) > 0:  # Verificar si la lista está vacía
                    centroidsNew.append(np.average(classifications[dot], axis=0))
                else:
                    centroidsNew.append(centroids[dot])  # Usar el centroide anterior si la lista está vacía
            e = [np.linalg.norm(np.array(new) - np.array(old)) for new, old in zip(centroidsNew, centroids)]
            if (np.abs(np.max(e)) <= self.tol) or (cont == self.max_iter):
                status = False
            else:
                cont += 1
                centroids = centroidsNew
        self.centroids = centroids
        return centroids

    def predict(self, info):
        distances = [info.resta(centroid) for centroid in self.centroids]
        classification = np.argmin(distances)
        return classification

def preprocess_image(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reducción de ruido
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Eliminar sombras
    dilated_img = cv2.dilate(denoised, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(denoised, bg_img)
    norm_img = diff_img.copy() # Normalizará la imagen en el rango 0-255
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Realzar contornos
    edges = cv2.Canny(thr_img, threshold1=30, threshold2=100)

    # Segmentación
    _, segmented = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Guardar la imagen segmentada para calcular la circularidad
    segmented_for_circularity = segmented.copy()

    # Normalizar la imagen
    normalized = cv2.normalize(segmented, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return normalized, segmented_for_circularity
# Funciones requeridas
def calculate_dominant_color(img):
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definir el umbral para considerar un pixel como blanco
    white_threshold = 200
    non_white_pixels = np.all(hsv < white_threshold, axis=2)

    # Split the HSV image into individual channels
    h, s, v = cv2.split(hsv)

    # Apply the non_white_pixels mask to each channel
    h = h[non_white_pixels]
    s = s[non_white_pixels]

    # Calcular el color dominante solo en los píxeles no blancos
    h = np.mean(h)
    s = np.mean(s)

    return h, s


# Calcular la circularidad de la imagen
def calculate_circularity(img):

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity * 10


# Calcular los momentos de Hu de la imagen
def calculate_hu_moments(img):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    # Seleccionar solo los momentos 1, 2, 3 y 6
    selected_hu_moments = [hu_moments[i][0] for i in [0, 1, 2, 5]]
    return selected_hu_moments*100


# Crear un conjunto de datos que incluya el color predominante, la circularidad y los momentos de Hu de cada imagen
# Direcciones de las imágenes
entrenamiento = r"C:\Users\tguev\Documents\Fing\IA\Curso\Fotos"

listTrain = os.listdir(entrenamiento)

# Parámetros
ancho, alto = 200, 200

data = []
centros = []
# Cargar imágenes de entrenamiento
for nameDir in listTrain:
    nombre = entrenamiento + "/" + nameDir  # Leemos las fotos
    for nameFile in os.listdir(nombre):  # asignamos etiquetas
        img = cv2.imread(os.path.join(nombre, nameFile))
        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        dominant_color = calculate_dominant_color(img)
        imgHu, imgCir = preprocess_image(img)
        circularity = calculate_circularity(imgCir)
        hu_moments = calculate_hu_moments(imgHu)
        # instanciar objeto del tipo foto
        newFoto = foto(dominant_color[0], dominant_color[1], circularity, hu_moments, nameFile)
        data.append(newFoto)
    centros.append(newFoto.car)

# Crear una instancia de KMeans
kmeans = KMeans(4)

# Ajustar los datos
centroides = kmeans.fit(data, centros)

# Clasificar las imágenes de entrenamiento
for i in data:
    print(i.id, kmeans.predict(i))

# Clasificar una imagen de prueba
img = cv2.imread(r"C:\Users\tguev\Documents\Fing\IA\Curso\test\Naranja7.jpg")
img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC)
dominant_color = calculate_dominant_color(img)
imgHu, imgCir = preprocess_image(img)
circularity = calculate_circularity(imgCir)
hu_moments = calculate_hu_moments(imgHu)
newFoto0 = foto(dominant_color[0], dominant_color[1], circularity, hu_moments)
print(kmeans.predict(newFoto0))









