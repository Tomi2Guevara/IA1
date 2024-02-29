import cv2
import numpy as np



# psr

class foto:
    def __init__(self, h, s, circularity, hu, id=None):
        self.id = id
        self.car = [h, s, circularity, hu[0]]

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

    def preprocess_image(self, image):
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
    def calculate_dominant_color(self, img):
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
    def calculate_circularity(self, img):

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    def calculate_hu_moments(self, img):
        moments = cv2.moments(img)
        hu_moments = cv2.HuMoments(moments)
        # Seleccionar solo los momentos 1, 2, 3 y 6
        selected_hu_moments = [hu_moments[i][0] for i in [0]] #con 5 es el recomendado
        return selected_hu_moments












