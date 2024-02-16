import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# psr

class foto:
    def __init__(self, h, s, circularity, hu, id=None):
        self.id = id
        self.car = [h, s, circularity]
        self.car.extend(hu)

    def resta(self, other):
        return np.linalg.norm(np.array(self.car) - other)


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
                    print(centroid)
                    distances = i.resta(centroid)

                classification = np.argmin(distances)
                classifications[classification].append(i.car)
            # creamos los nuevos centroides
            centroidsNew = []
            for dot in range(len(classifications)):
                centroidsNew[dot] = np.average(classifications[dot], axis=0)
            e = np.abs(centroidsNew - centroids)
            if (np.abs(np.max(e)) <= self.tol) or (cont == self.max_iter):
                status = False
            else:
                cont += 1
                centroids = centroidsNew
        self.centroids = centroids
        return centroids

    def predict(self, data):
        distances = [np.linalg.norm(data.car - centroid) for centroid in self.centroids]
        classification = np.argmin(distances)
        return classification


# Funciones requeridas
def calculate_dominant_color(img):
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    h = np.mean(h)
    s = np.mean(s)

    return h, s


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
    return circularity * 100


# Calcular los momentos de Hu de la imagen
def calculate_hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments



