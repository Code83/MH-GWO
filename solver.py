from pickletools import int4
import numpy as np
import leerProblema as Instance
import time
import MH
import leerProblema as Instance
import factibilidad as fc
import fitness as fit
import reparacion as rep
import discretizacion

# tomo el tiempo inicial para la ejecucion completa
tiempoInicial = time.time()

# ruta que deben modificar dependiendo de donde estén sus instancias
dirSCP = 'SCP/scp510.txt'
dirResultado = 'Resultados/'

print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("problema resuelto: "+dirSCP.split("/")[1].split(".")[0])

# metaheuristica a utilizar
# esta Grey Wolf Optimizer (GWO) y Sine Cosine Algorithm (SCA)
mh = "GWO"
problema = dirSCP.split("/")[1].split(".")[0]

resultado = open(dirResultado+mh+"_"+problema+".txt", "w")


# lectura de la instancia
SCP = Instance.read_instance(dirSCP)

# matriz de cobertura m x n
matrizCobertura = SCP["S"]    
# vector de costos para cada columna n
vectorCostos = SCP["C_s"]

# obtengo las dimensiones de mi problema (m)
dim = len(vectorCostos)
# tamaño de la poblacion 
pob = 25
# numero de iteraciones a iterar
maxIter = 200
# inicializo un diccionario
params = {}

DS = ['S4','Elitist'] #[v1,Standard]
#DS = ['v1','Standard']
#DS = ['S1','Elitist']
#DS = ['S4', 'Standard']


# guardo en mi diccionario todos las cosas que necesito para iterar, costos, matriz de cobertura, como voy a binarizar y como voy a reparar las soluciones infactibles
params["costos"] = vectorCostos
params["cobertura"] = matrizCobertura
params["ds"] = DS
params["repairType"] = 1

# Generar población inicial
poblacion = np.random.uniform(low=-10.0, high=10.0, size=(pob,dim))

# Genero una población inicial binaria, esto ya que nuestro problema es binario (SCP)
matrixBin = np.random.randint(low=0, high=2, size = (pob,dim))
# Genero un vector donde almacenaré los fitness de cada individuo
fitness = np.zeros(pob)

# Genero un vetor dedonde tendré mis soluciones rankeadas
solutionsRanking = np.zeros(pob)

# calculo de factibilidad de cada individuo y calculo del fitness inicial
for i in range(poblacion.__len__()):
    flag, aux = fc.check_sol(SCP,matrixBin[i])
    if not flag: #solucion infactible
        # matrixBin[i] = rep.reparaSimple(SCP, matrixBin[i])
        matrixBin[i] = rep.reparaComplejo(SCP, matrixBin[i])
        

    fitness[i] = fit.fitness(SCP, matrixBin[i])  

solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
bestRowAux = solutionsRanking[0]
# mostramos nuestro fitness iniciales
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("fitness incial: "+str(fitness))
print("Best fitness inicial: "+str(np.min(fitness)))
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh)
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
# for de iteraciones
for iter in range(0, maxIter):
    
    # obtengo mi tiempo inicial
    processTime = time.process_time()  
    timerStart = time.time()


    # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    Best = poblacion[bestRowAux]
    BestBinary = matrixBin[bestRowAux]
    BestFitness = np.min(fitness)
    
    # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
    # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones

    if mh == "SCA":
        poblacion = MH.iterarSCA(maxIter, iter, dim, poblacion.tolist(), Best.tolist())
    if mh == "GWO":
        poblacion = MH.iterarGWO(maxIter, iter, dim, poblacion.tolist(), fitness.tolist())
    
    # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
    for i in range(poblacion.__len__()):
        ds = discretizacion.DiscretizationScheme(poblacion,matrixBin,solutionsRanking,DS[0],DS[1])

        matrixBin = ds.binariza()

        flag, aux = fc.check_sol(SCP,matrixBin[i])
        if not flag: #solucion infactible
            # matrixBin[i] = rep.reparaSimple(SCP, matrixBin[i])
            matrixBin[i] = rep.reparaComplejo(SCP, matrixBin[i])
            

        fitness[i] = fit.fitness(SCP, matrixBin[i])

        # #Conservo el Best
        # if fitness[i] < BestFitness:
        #     BestFitness = fitness[i]
        #     BestBinary = matrixBin[i]
        #     bestRowAux = i
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes

    #Conservo el Best
    if fitness[bestRowAux] > BestFitness:
        fitness[bestRowAux] = BestFitness
        matrixBin[bestRowAux] = BestBinary
    BestFitnes = np.min(fitness)

    timerFinal = time.time()
    # calculo mi tiempo para la iteracion t
    timeEjecuted = timerFinal - timerStart
    print("iteracion: "+str(iter)+", best fitness: "+str(BestFitness)+", tiempo iteracion (s): "+str(timeEjecuted))
    resultado.write("iteracion: "+str(iter)+", best fitness: "+str(BestFitness)+", tiempo iteracion (s): "+str(timeEjecuted)+"\n")

print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Best fitness: "+str(BestFitness))
resultado.write("Best fitness: "+str(BestFitness)+"\n")
print("Cantidad de columnas seleccionadas: "+str(sum(BestBinary)))
resultado.write("Cantidad de columnas seleccionadas: "+str(sum(BestBinary))+"\n")
print("Best solucion: \n"+str(BestBinary.tolist()))
resultado.write("Best solucion: \n"+str(BestBinary.tolist())+"\n")
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
tiempoFinal = time.time()
tiempoEjecucion = tiempoFinal - tiempoInicial
print("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
resultado.write("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
resultado.close()
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")