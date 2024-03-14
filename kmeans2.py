import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
class foto:
    def __init__(self, circularity, hu, hist, id=None):
        self.id = id
        self.car = [circularity, hu[0]**2]
        for i in range(3):
           self.car.append(hist[i][0])

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
                distances = []
                for centroid in centroids:
                    distances.append(i.resta(centroid))

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
        distances = []
        for i in self.centroids:
            distances.append(info.resta(i))

        return np.argmin(distances)


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
        selected_hu_moments = [hu_moments[i][0] for i in [1]] #con 5 es el recomendado
        return selected_hu_moments

    def graficar(self, data, km):
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
        centroides_pca = pca.transform(self.centroids)

        # Crea una figura 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Traza los datos
        for i in range(len(data_pca)):
            ax.scatter(data_pca[i, 0], data_pca[i, 1], data_pca[i, 2], color=colores[km.predict(data[i])])

        # Traza los centroides
        for i in range(len(centroides_pca)):
            ax.scatter(centroides_pca[i, 0], centroides_pca[i, 1], centroides_pca[i, 2], color=colores[i], s=100,
                       marker='x')

        plt.show()

    def visualizar(self, normalized_image, segmented_for_circularity):

        # Mostrar la imagen normalizada
        plt.figure(figsize=(10, 10))
        plt.imshow(normalized_image, cmap='gray')
        plt.title('Normalized Image')
        plt.show()

        # Mostrar la imagen segmentada para calcular la circularidad
        plt.figure(figsize=(10, 10))
        plt.imshow(segmented_for_circularity, cmap='gray')
        plt.title('Segmented Image for Circularity Calculation')
        plt.show()

    def saveCents(self):

        with open('centroids.pkl', 'wb') as f:
            pickle.dump(self.centroids, f)

    def loadCents(self):

        with open('centroids.pkl', 'rb') as f:
            self.centroids = pickle.load(f)

        return self.centroids

    def photo(self, link):
        img = cv2.imread(link)
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
        imgHu, imgCir = self.preprocess_image(img)
        circularity = self.calculate_circularity(imgCir)
        hu_moments = self.calculate_hu_moments(imgHu)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        return (circularity * 10) ** 2, (hu_moments * 10), hist

    def recImg(self,image_name, prediction):
        # Cargamos la imagen
        image = cv2.imread(image_name)

        # Definimos los colores para cada clase en formato BGR
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        # Convertimos la imagen a escala de grises y aplicamos un umbral
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Encontramos los contornos en la imagen
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujamos un rectángulo alrededor de cada contorno
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:
                x, y, width, height = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + width, y + height), colors[prediction], 3)

        # Mostramos la imagen
        pick = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(pick)
        plt.show()













