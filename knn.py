import librosa
import numpy as np
import pickle

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
                max_len = max(len(features1), len(features2))
                features1_padded = np.pad(features1, (0, max_len - len(features1)))
                features2_padded = np.pad(features2, (0, max_len - len(features2)))

                dist.append(np.linalg.norm(features1_padded - features2_padded))
            normas.append(np.linalg.norm(dist))
        return normas



    def preprocess_audio(self, file_path):
        # Cargar el audio
        y, sr = librosa.load(file_path, sr=None)

        # Usar trim para cortar las partes del audio que no sirven
        y, _ = librosa.effects.trim(y, top_db=17)

        # Preprocesar los audios (eliminar ruidos de fondo)
        y = librosa.effects.percussive(y)

        y = np.array(y)

        # Normalizar los audios
        y = librosa.util.normalize(y)

        # Dividir el audio en diferentes sectores (opcional)
        #frames = librosa.util.frame(y)


        # Seleccionar los momentos más relevantes del audio
        #mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        return y, sr

    def audio(self, link):
        y, sr = self.preprocess_audio(link)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)[0]
        zcr = librosa.feature.zero_crossing_rate(y,  hop_length=int(len(y)/10))[0]
        duration = len(y) / sr

        dataTest = []
        for i in mfccs:
            dataTest.append(i)
        for i in zcr:
            dataTest.append(i)
        dataTest.append(duration)

        return dataTest


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
    #crear un método para guardar xTrain y yTrain


    def save_data(self):
        with open('xTrain.pkl', 'wb') as f:
            pickle.dump(self.X_train, f)

        with open('yTrain.pkl', 'wb') as f:
            pickle.dump(self.y_train, f)

    def load_data(self):
        with open('xTrain.pkl', 'rb') as f:
            self.X_train = pickle.load(f)

        with open('yTrain.pkl', 'rb') as f:
            self.y_train = pickle.load(f)







