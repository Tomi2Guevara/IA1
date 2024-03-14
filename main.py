from kmeans2 import KMeans
from kmeans2 import foto
from knn import KNN
import os



status = True
km = KMeans(4)
knn = KNN(3)
while status:

    print("Menú")
    print("1. Entrenar")
    print("2. Cargar")
    print("3. graficar centroides")
    print("4. guardar datos")
    print("5. Salir")
    opcion = int(input("Ingrese una opción: "))

    fotos = r"C:\Users\tguev\Documents\Fing\IA\Curso\test\Fotos"
    test = r"C:\Users\tguev\Documents\Fing\IA\Curso\test\audio"


    if opcion == 1:
        # ----------------------- FOTOS -----------------------

        # Crear un conjunto de datos que incluya el color predominante, la circularidad y los momentos de Hu de cada imagen
        # Direcciones de las imágenes
        entrenamiento = r"C:\Users\tguev\Documents\Fing\IA\Curso\Fotos"

        listTrain = os.listdir(entrenamiento)

        # Parámetros
        #ancho, alto = 200, 200


        data = []
        centros = []

        # Cargar imágenes de entrenamiento
        for nameDir in listTrain:
            nombre = entrenamiento + "/" + nameDir  # Leemos las fotos
            for nameFile in os.listdir(nombre):  # asignamos etiquetas
                file_path = os.path.join(nombre, nameFile)
                circularity, hu_moments, hist= km.photo(file_path)
                # instanciar objeto del tipo foto
                newFoto = foto(circularity**1.1, hu_moments, hist, nameFile)
                cadena = '\t'.join([f'{v:.2f}'.replace('.', ',') for v in newFoto.car])
                print(cadena)
                data.append(newFoto)
            print('-------------------')
            centros.append(newFoto.car)

        # Ajustar los datos
        centroides = km.fit(data, centros)


        #clasificar los datos de entrenamiento
        for i in data:
            print(i.id, km.predict(i))

        #for i in centroides:
            #cadena = '\t'.join([f'{v:.2f}'.replace('.', ',') for v in i])
            #print(cadena)

        # ----------------------- AUDIO -----------------------

        # Cargar y preprocesar los audios
        entrenamiento = r"C:\Users\tguev\Documents\Fing\IA\Curso\audios"

        listTrain = os.listdir(entrenamiento)


        xTrain = []
        yTrain = []
        # Cargar audios de entrenamiento
        for nameDir in listTrain:
            nombre = entrenamiento + "//" + nameDir  # Leemos las fotos
            for nameFile in os.listdir(nombre):  # asignamos etiquetas
                file_path = os.path.join(nombre, nameFile)
                sound = knn.audio(file_path)
                xTrain.append(sound)
                yTrain.append(knn.label_for_filename(nombre))

        # Ajustar los datos
        knn.fit(xTrain, yTrain)



    elif opcion == 2:

        try:
            centroides = centroides

        except:
            # cargar los datos
            centroides = km.loadCents()

        # clasificar las fotos

        listTrain = os.listdir(fotos)
        fotoPred = []
        paths = []
        newFotoData = []
        for nameDir in listTrain:
            nombre = fotos + "/" + nameDir  # Leemos las fotos
            file_path = os.path.join(nombre)
            circularity, hu_moments, hist = km.photo(file_path)
            newFoto0 = foto(circularity, hu_moments, hist, nameDir)
            newFotoData.append(newFoto0)
            fotoPred.append(km.predict(newFoto0))
            paths.append(file_path)

        for i in range(len(fotoPred)):
            print(newFotoData[i].car)
            print(fotoPred[i])
        for i in centroides:
            cadena = '\t'.join([f'{v:.2f}'.replace('.', ',') for v in i])
            print(cadena)




        # ----------------------- USO Audio -----------------------
        try:
            xTrain = xTrain
            yTrain = yTrain
        except:
            # cargar los datos
            knn.load_data()


        # cargamos el audio a identificar
        testList = os.listdir(test)
        test = test + "\\" + testList[0]
        dataTest = knn.audio(test)
        comando = knn.predict(dataTest)

        for i in fotoPred:

            if i == comando:
                km.recImg(paths[fotoPred.index(i)], comando)

        # graficar(newFotoData, centroides, km)
        # for i in centroides:
        # cadena = '\t'.join([f'{v:.2f}'.replace('.', ',') for v in i])
        # print(cadena)


    elif opcion == 3:
        km.graficar(data, km)

    elif opcion == 4:
        km.saveCents()
        knn.save_data()
        print("Datos guardados")

    elif opcion == 5:
        status = False

    else:
        print("Opción no válida")