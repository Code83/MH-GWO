import numpy as np

def read_instance(Instancia):

    file = open(Instancia, "r")

    ### Genero una lista con todos los valores del archivo. Se almacenan en u.
    lines = file.readlines()
    lines = [line.replace('\n', ' ').strip().split(" ") for line in lines]
    u = []
    for item in lines:
        u += item
    u = list(map(int, u))

    # Leer Dimensión
    # Con la función pop() retorno el primer elemento de la lista u, y se elimina de la lista.
    m = u.pop(0)
    n = u.pop(0)
    # print("Número de restricciones: "+str(m))
    # print("Número de columnas: "+str(n))
    # print("Dimensiones de la matriz de cobertura: ["+str(m)+" , "+str(n)+"]")

    
    # Leer Costo
    # Obtengo los costos de cada subset (columna)

    C = [int(i) for i in u[:n]]
    Cnp = np.array(C)
    # print("Costos para cada columna: "+str(Cnp.tolist()))
    # print("Cantidad de costos almacenados: "+str(Cnp.__len__()))

    # Elimino la lista de costos de u
    u = u[n:]

    # Verifico la correcta lectura de las variables
    # print(n, m, len(C), u[0])

    # Leer Restricciones
    SS = []
    c_s = []
    set = np.zeros((m,n))
    for j in range(m):
        # Obtengo el tamaño del subset
        aux = u.pop(0)
        # print("Cantidad de columnas que cubre la restriccion "+str(j)+": "+str(aux))

        # Capturo el subset y le resto 1 a cada valor, para usarlo como índice
        s = u[:aux]
        s = list(map(lambda elemento:elemento-1, s))
        # print("Columnas que cumbren la restriccion "+str(j)+": "+str(s))

        # Actualizo cobertura, coloco los 1 en las posiciones correspondientes
        # print("Antes de colocar unos: "+str(set[j,s].tolist()))
        set[j,s] = 1
        # print("Despues de colocar unos: "+str(set[j,s].tolist()))

        # Arego a la lista de subsets
        SS.append(s)

        # Elimino el subset de u
        u = u[aux:]

    # print(set)

    # validacion de colocacion de 1 
    # print("Validacion de matriz de cobertura: "+str(set[0][90]))

    SCP        = {}
    SCP["n"]   = n          # Numero de subsets / coberturas / columnas
    SCP["m"]   = m          # Numero de restricciones / zonas
    SCP["C_s"] = Cnp        # Costo de seleccionar cada subset / cobertura
    SCP["S"]   = set        # Subsets de la instancia - matriz de cobertura

    return SCP