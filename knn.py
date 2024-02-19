import librosa
import os
import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        return self._predict(X[0])

    def _predict(self, x):
        distances = euclidean_distance(x, self.X_train)
        ys = []
        k_indices = np.argsort(distances)[:self.k]
        for i in range(self.k):
            print(self.y_train[k_indices[i]])
            ys.append(self.y_train[k_indices[i]])

        return max(ys, key=ys.count)


def label_for_filename(filename):
    if "banana" in filename:
        return 0
    elif 'manzana' in filename:
        return 1
    elif 'pera' in filename:
        return 2
    elif 'naranja' in filename:
        return 3
    else:
        raise ValueError("Unknown filename: {}".format(filename))


# Cargar y preprocesar los audios
entrenamiento = r"C:/Users/tguev\Documents\Fing\IA\Curso/redesNeuronales\DataSet\dataset/audios\processed"

listTrain = os.listdir(entrenamiento)

xTrain = []
yTrain = []
# Cargar imágenes de entrenamiento
for nameDir in listTrain:
    nombre = entrenamiento + "//" + nameDir  # Leemos las fotos
    for nameFile in os.listdir(nombre):  # asignamos etiquetas
        file_path = os.path.join(nombre, nameFile)
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features = np.concatenate((mfccs, chroma, contrast, tonnetz), axis=0).T
        xTrain.append(features)
        yTrain.append(label_for_filename(nombre))

# Crear una instancia de KNN
knn = KNN(k=5)

# Ajustar los datos (asumiendo que 'y' es tu vector de etiquetas)
knn.fit(features, yTrain)

test = r"C:\Users\tguev\Documents\Fing\IA\Curso\redesNeuronales\DataSet\dataset\audios\processed\pera\pera2.wav"
y, sr = librosa.load(test, sr=None)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
features = np.concatenate((mfccs, chroma, contrast, tonnetz), axis=0).T
print('predict:',knn.predict(features))
print(features.shape)


