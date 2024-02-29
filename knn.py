import librosa
import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        distances = self.euclidean_distance(x)
        ys = []
        k_indices = np.argsort(distances)[:self.k]
        for i in range(self.k):
            print(self.y_train[k_indices[i]])
            ys.append(self.y_train[k_indices[i]])

        return max(ys, key=ys.count)

    def euclidean_distance(self, features1):

        normas = []
        for j in self.X_train:
            features2 = j
            dist = []

            for i in range(len(features1)):
                # Asegurarse de que los vectores tienen la misma longitud
                max_len = max(len(features1[i]), len(features2[i]))
                features1_padded = np.pad(features1[i], (0, max_len - len(features1[i])))
                features2_padded = np.pad(features2[i], (0, max_len - len(features2[i])))

                dist.append(np.linalg.norm(features1_padded - features2_padded))
            normas.append(np.linalg.norm(dist))
        return normas



    def preprocess_audio(self, file_path):
        # Cargar el audio
        y, sr = librosa.load(file_path, sr=None)

        # Usar trim para cortar las partes del audio que no sirven
        y, _ = librosa.effects.trim(y)

        # Preprocesar los audios (eliminar ruidos de fondo)
        y = librosa.effects.percussive(y)

        # Normalizar los audios
        y = librosa.util.normalize(y)

        # Dividir el audio en diferentes sectores (opcional)
        #frames = librosa.util.frame(y)


        # Seleccionar los momentos más relevantes del audio
        #mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        return y, sr
    def label_for_filename(self, filename):
        if "Banana" in filename:
            return 0
        elif 'Manzana' in filename:
            return 1
        elif 'Pera' in filename:
            return 3
        elif 'Naranja' in filename:
            return 2
        else:
            raise ValueError("Unknown filename: {}".format(filename))





