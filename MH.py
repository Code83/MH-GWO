from ctypes import util
import random
import math
import numpy as np

#---------------------------------------------------------------------------SINE COSINE ALGORITHM---------------------------------------------------------------------------------#
# esta funcion recibe como parametros:
# total de iteraciones: maxIter
# iteracion actual: t
# dimensiones del problema: dimension
# la poblacion completa CONTINUA: poblacion
# mejor individuo de la poblacion: bestSolutionCon
def iterarSCA(maxIter, t, dimension, poblacion, bestSolutionCon):
    # nuestro valor de a es constante y es extraido desde paper
    a = 2
    # calculamos nuestro r1
    r1 = a - (t * (a / maxIter))
    # for de individuos
    for i in range(poblacion.__len__()):
        # for de dimensiones
        for j in range(dimension):
            # calculo un numero aleatoreo entre [0,1]
            rand = random.uniform(0.0, 1.0)
            # calculo r2
            r2 =  (2 * math.pi) * rand 
            # calculo r3
            r3 = 2 * rand
            # calculo r4
            r4 = random.uniform(0.0, 1.0)
            if r4 < 0.5:
                # perturbo la poblacion utilizando como base la funcion seno
                poblacion[i][j] = poblacion[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolutionCon[j] ) - poblacion[i][j] ) ) )
            else:
                # perturbo la poblacion utilizando como base la funcion coseno
                poblacion[i][j] = poblacion[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolutionCon[j] ) - poblacion[i][j] ) ) )
    # retorno la poblacion modificada 
    return np.array(poblacion)



#---------------------------------------------------------------------------GREY WOLF OPTIMIZER---------------------------------------------------------------------------------#
# esta funcion recibe como parametros:
# total de iteraciones: maxIter
# iteracion actual: t
# dimensiones del problema: dimension
# la poblacion completa CONTINUA: poblacion
# todos los fitness de los individuos: fitness
def iterarGWO(maxIter, t, dimension, poblacion, fitness):
    # nuestro parametro a que decrece linealmente de 2 a 0
    a = 2 - t * ((2) / maxIter) 
    # esta funcion nos devuelve una lista ordenada por fitness con los mejores individuos
    #posicionesOrdenadas = util.selectionSort(fitness)
    posicionesOrdenadas = np.argsort(fitness)

    # defino mi lobo alfa
    Xalfa  = poblacion[posicionesOrdenadas[0]]
    # defino mi lobo beta
    Xbeta  = poblacion[posicionesOrdenadas[1]]
    # defino mi lobo delta
    Xdelta = poblacion[posicionesOrdenadas[2]]

    # for de individuos
    for i in range(poblacion.__len__()):
        # for de dimensiones
        for j in range(dimension):

            # calculamos r1 que es un numero aleatorea en [0,1]
            r1 = random.random()  
            # calculamos r2 que es un numero aleatorea en [0,1]
            r2 = random.random()  
            # Calculamos A1
            A1 = 2 * a * r1 - a
            # Calculamos C1
            C1 = 2 * r2
            # Calculamos D para nuestro lobo alfa
            dalfa = abs((C1*Xalfa[j])-poblacion[i][j])
            # Calculamos X1
            X1 = Xalfa[j] - (A1 * dalfa)
            
            # calculamos r1 que es un numero aleatorea en [0,1]
            r1 = random.random()  
            # calculamos r2 que es un numero aleatorea en [0,1]
            r2 = random.random()  
            # Calculamos A2
            A2 = 2 * a * r1 - a
            # Calculamos C2
            C2 = 2 * r2
            # Calculamos D para nuestro lobo beta
            dbeta = abs((C2*Xbeta[j])-poblacion[i][j])
            # Calculamos X2
            X2 = Xbeta[j] - (A2 * dbeta)

            # calculamos r1 que es un numero aleatorea en [0,1]
            r1 = random.random()  
            # calculamos r2 que es un numero aleatorea en [0,1]
            r2 = random.random()  
            # Calculamos A3
            A3 = 2 * a * r1 - a
            # Calculamos C3
            C3 = 2 * r2
            # Calculamos D para nuestro lobo delta
            ddelta = abs((C3*Xdelta[j])-poblacion[i][j])
            # Calculamos X3
            X3 = Xdelta[j] - (A3 * ddelta) 

            # perturbamos al individuo i en su dimension j
            poblacion[i][j] = (X1+X2+X3)/3

    # retornamos la poblacion modificada
    return np.array(poblacion)

