import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
#import librosa

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = [np.linalg.norm(x - self.X_train, axis=1) for x in X]
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = []
        for i in range(len(k_nearest_labels)):
            most_common.append(k_nearest_labels[i].max())

        
        return most_common
    
def label_for_filename(filename):
    if "banana" in filename:
        return 0
    elif  'manzana' in filename:
        return 1
    elif 'pera' in filename:
        return 2
    elif 'naranja' in filename:
        return 3
    else:
        raise ValueError("Unknown filename: {}".format(filename))

# Cargar y preprocesar los audios

entrenamiento = "C:/Users/tguev/Documents/Fing/IA/Proyecto/Curso/redesNeuronales/DataSet/dataset/audios/original"
prueba = "C:/Users/tguev/Documents\Fing\IA\Proyecto\Curso/redesNeuronales\DataSet\dataset/audios\processed"

listTrain = os.listdir(entrenamiento)
listTest = os.listdir(prueba)

Xtrain = []
ytrain = []

Xtest = []
ytest = []

for nameDir in listTrain:
    dir_path = os.path.join(entrenamiento, nameDir)
    for filename in os.listdir(dir_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(dir_path, filename)
            try:
                rate, signal = wav.read(file_path)
                mfcc_features = mfcc(signal, rate)
                Xtrain.append(mfcc_features.mean(axis=0))
                ytrain.append(label_for_filename(filename))
            except ValueError as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"Skipping {filename}")

for nameDir in listTest:
    dir_path = os.path.join(prueba, nameDir)
    for filename in os.listdir(dir_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(dir_path, filename)
            try:
                rate, signal = wav.read(file_path)
                mfcc_features = mfcc(signal, rate)
                Xtest.append(mfcc_features.mean(axis=0))
                ytest.append(label_for_filename(filename))
            except ValueError as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"Skipping {filename}")

# Convertir a arrays de numpy para facilitar el manejo de los datos
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
Xtest = np.array(Xtest)
ytest = np.array(ytest)



# Entrenar y probar el algoritmo KNN
knn = KNN(k=50)
knn.fit(Xtrain, ytrain)
#print(X_test)
predictions = knn.predict(Xtest)


print("Predicciones: ", predictions)
print("Etiquetas verdaderas: ", ytest)
print(len(predictions) == len(ytest))