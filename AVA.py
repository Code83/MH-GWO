# -*- coding: utf-8 -*-
"""AVA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FvjnJMbL5dXF5DrR1ddXzKn3Mv0wODAK
"""

# maili id - spaerixinfotech@gmail.com

def check_sol(SCP, sol):
    flag = True
    set = SCP["S"]
    aux = np.dot(set,sol)           # Multiplico la matriz de cobertura por la solucion

    if 0 in aux:                    # Si alguna zona no esta cubierta, retorno False
        flag = False

    return flag, aux

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


import discretizacion

SCP = {}

def loadProblem(fileLoad):
  dirSCP = 'SCP/'+fileLoad+'.txt'

  SCP = read_instance(dirSCP)
  return SCP

def reparaComplejo(SCP, solution):
    sol =  np.reshape(solution, (SCP["n"],))
    set = SCP["S"]
    
    feasible, aux = check_sol(SCP,sol)

    while not feasible:
        r_no_cubiertas = np.zeros((SCP["m"],))
        r_no_cubiertas[np.argwhere(aux == 0)] = 1           # Vector indica las restricciones no cubiertas
        cnc = np.dot(r_no_cubiertas, set)                   # Cantidad de restricciones no cubiertas que cubre cada columna (de tamaño n)
        trade_off = np.divide(SCP["C_s"],cnc)               # Trade off entre zonas no cubiertas y costo de seleccionar cada columna
        idx = np.argmin(trade_off)                          # Selecciono la columna con el trade off mas bajo
        sol[idx] = 1                                        # Asigno 1 a esa columna
        feasible, aux = check_sol(SCP,sol)               # Verifico si la solucion actualizada es factible

    return sol

import random
import time
import numpy as np
rng = np.random.default_rng()
import math
import sys
from numpy import linalg as LA
import numpy as np
import math

def fun(X):
    #output = sum(np.square(X))+random.random()
    cost = SCP["C_s"]
    output = np.dot(X, cost)
    return output

# This function is to initialize the Vulture population.
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
    return X

# Calculate fitness values for each Vulture
def CalculateFitness(X,fun):
    fitness = fun(X)
  
    return fitness

# Sort fitness.
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index


# Sort the position of the Vulture according to fitness.
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew


# Boundary detection function.
def BorderCheck1(X,lb,ub,dim):
        for j in range(dim):
            if X[j]<lb[j]:
                X[j] = ub[j]
            elif X[j]>ub[j]:
                X[j] = lb[j]
        return X

def rouletteWheelSelection(x):
    CS  = np.cumsum(x)
    Random_value = random.random()
    index = np.where(Random_value <= CS)
    index = sum(index)
    return index

def random_select(Pbest_Vulture_1,Pbest_Vulture_2,alpha,betha):
    probabilities=[alpha, betha ]
    index = rouletteWheelSelection( probabilities )
    if ( index.all()> 0):
            random_vulture_X=Pbest_Vulture_1
    else:
            random_vulture_X=Pbest_Vulture_2
    
    return random_vulture_X

def exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound):
    if random.random()<p1:
        current_vulture_X=random_vulture_X-(abs((2*random.random())*random_vulture_X-current_vulture_X))*F;
    else:
        current_vulture_X=(random_vulture_X-(F)+random.random()*((upper_bound-lower_bound)*random.random()+lower_bound));
    return current_vulture_X

def exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X,random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound):
    if  abs(F)<0.5:
        
        if random.random()<p2:
            
            A=Best_vulture1_X-((np.multiply(Best_vulture1_X,current_vulture_X))/(Best_vulture1_X-current_vulture_X**2))*F
            B=Best_vulture2_X-((Best_vulture2_X*current_vulture_X)/(Best_vulture2_X-current_vulture_X**2))*F
            current_vulture_X=(A+B)/2
        else:
            current_vulture_X=random_vulture_X-abs(random_vulture_X-current_vulture_X)*F*levyFlight(variables_no)
            
    if random.random()>=0.5:
        if random.random()<p3:
            current_vulture_X=(abs((2*random.random())*random_vulture_X-current_vulture_X))*(F+random.random())-(random_vulture_X-current_vulture_X)
            
        else:
            s1=random_vulture_X*(random.random()*current_vulture_X/(2*math.pi))*np.cos(current_vulture_X)
            s2=random_vulture_X*(random.random()*current_vulture_X/(2*math.pi))*np.sin(current_vulture_X)
            current_vulture_X=random_vulture_X-(s1+s2)
    return current_vulture_X

# eq (18) 
def levyFlight(d):  
    beta=3/2;
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u=np.random.randn(1,d)*sigma;
    v=np.random.randn(1,d);
    step=u/abs(v)**(1/beta);
    o=step;
    return o

def AVA(pop,dim,lb,ub,Max_iter,fun, DS):

    matrixBin = np.random.randint(low=0, high=2, size = (pop,dim))
    # Genero un vector de donde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)

    alpha=0.8
    betha=0.2
    p1 = 0.6
    p2=0.4
    p3=0.6
    Gama = 2.5
    X = initial(pop, dim, lb,ub)                    # Initialize the random population 
    fitness = np.zeros([pop, 1])
    for i in range(pop):
      ds = discretizacion.DiscretizationScheme(X, matrixBin,solutionsRanking,DS[0],DS[1])

      matrixBin = ds.binariza()

      flag, aux = check_sol(SCP,matrixBin[i])

      if not flag: #solucion infactible
        matrixBin[i] = reparaComplejo(SCP, matrixBin[i])

      #fitness[i] = CalculateFitness(X[i, :], fun, matrixBin[i, :], solutionsRanking,DS)
      fitness[i] = CalculateFitness(matrixBin[i, :], fun)

    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes

    fitness, sortIndex = SortFitness(fitness)       # Sort the fitness values of African Vultures
    X = SortPosition(X, sortIndex)                  # Sort the African Vultures population based on fitness
    GbestScore = fitness[0]                         # Stores the optimal value for the current iteration.
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[0, :]

    GbestPositonBin = np.zeros([1, dim])
    GbestPositonBin[0, :] = matrixBin[0, :]

    Curve = np.zeros([Max_iter, 1])
    Xnew = np.zeros([pop, dim])
    XBinnew = np.zeros([pop, dim])
    # Main iteration starts here
    for t in range(Max_iter):                       
      # obtengo mi tiempo inicial
      processTime = time.process_time()  
      timerStart = time.time() 

      Pbest_Vulture_1  = X[0,:]                     #location of Vulture (First best location Best Vulture Category 1) 
      Pbest_Vulture_2  = X[1,:]                     #location of Vulture (Second best location Best Vulture Category 1)
      t3=np.random.uniform(-2,2,1)*((np.sin((math.pi/2)*(t/Max_iter))**Gama)+np.cos((math.pi/2)*(t/Max_iter))-1)
      z = random.randint(-1, 0)
      #F= (2*random.random()+1)*z*(1-(t/Max_iter))+t3
      P1=(2*random.random()+1)*(1-(t/Max_iter))+t3
      F=P1*(2*random.random()-1)
      
      # For each vulture Pi
      for i in range(pop):
        current_vulture_X = X[i,:]
        random_vulture_X=random_select(Pbest_Vulture_1,Pbest_Vulture_2,alpha,betha)   # select random vulture using eq(1)
        if abs(F) >=1:
          current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, ub, lb) # eq (16) & (17)
          
        else:
          current_vulture_X = exploitation(current_vulture_X, Pbest_Vulture_1, Pbest_Vulture_2, random_vulture_X, F, p2, p3, dim, ub, lb) # eq (10) & (13)

        Xnew[i,:] = current_vulture_X[0]
        Xnew[i,:] = BorderCheck1(Xnew[i,:], lb, ub, dim)

        ds = discretizacion.DiscretizationScheme(Xnew, matrixBin,solutionsRanking,DS[0],DS[1])

        XBinnew = ds.binariza()

        flag, aux = check_sol(SCP,XBinnew[i])

        if not flag: #solucion infactible
          XBinnew[i] = reparaComplejo(SCP, XBinnew[i])
          
        #tempFitness = CalculateFitness(Xnew[i,:], fun, matrixBin[i,:], solutionsRanking,DS)
        tempFitness = CalculateFitness(XBinnew[i,:], fun)
        # Update local best solution
        if (tempFitness <= fitness[i]):
            fitness[i] = tempFitness               
            X[i,:] = Xnew[i,:] 
            matrixBin[i,:] = XBinnew[i,:] 

      Ybest,index = SortFitness(fitness) 
      X = SortPosition(X, index)
      matrixBin = SortPosition(matrixBin, index)

      # Update global best solution
      if (Ybest[0] <= GbestScore): 
        GbestScore = Ybest[0]
        GbestPositon[0, :] = X[index[0], :]
        GbestPositonBin[0, :] = matrixBin[index[0], :]

      timerFinal = time.time()
      timeEjecuted = timerFinal - timerStart
      #print("iteracion: "+str(t)+", best fitness: "+str(GbestScore)+", tiempo iteracion (s): "+str(timeEjecuted))

      #print(GbestPositon)
      #print(GbestPositonBin)
      Curve[t] = GbestScore

      solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness

    validateFit = CalculateFitness(GbestPositonBin, fun)
    print('Fitness transformado:', validateFit)
    
    return Curve,GbestPositon,GbestScore,GbestPositonBin

import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

#DS = ['v1','Standard']
DS = ['S2','Elitist']

listProb = ['scp41', 'scp42', 'scp51', 'scp52', 'scp63']

for problem in listProb:
    #problem = 'scp41'
    SCP = loadProblem(problem)

    print('---------------------------------------------------------')
    print('Problema a resolver: ' + problem)

    rng = np.random.default_rng()

    pop = 10                     # Population size n
    MaxIter = 100                # Maximum number of iterations.
    #dim = 30                    # The dimension.

    vectorCostos = SCP["C_s"]
    dim = len(vectorCostos)

    fl=-100                      # The lower bound of the search interval.
    ul=100                      # The upper bound of the search interval.

    ejecNumber = 3
    fitnessEjec = np.zeros([ejecNumber, 1])

    CurveArr = []

    for f in range(ejecNumber):
        time_start = time.time()

        lb = fl*np.ones([dim, 1])
        ub = ul*np.ones([dim, 1])

        Curve,GbestPositon,GbestScore,GbestPositonBin = AVA(pop, dim, lb, ub, MaxIter, fun, DS) # Afican Vulture Optimization Algorithm
        time_end = time.time()
        print(f"Tiempo ejecución: {time_end  - time_start } s")
        print('Valor optimo：',GbestScore)
        ##print('Solución optima real：',GbestPositon)
        #print('Solución Optima：',GbestPositonBin[0])
        print("Cantidad de columnas seleccionadas: "+str(sum(GbestPositonBin[0])))

        CurveArr.append(Curve)

        fitnessEjec[f] = GbestScore

    #print('Fitness encontrados:', str(fitnessEjec))
    print('###############################')
    print('Problema: ' + problem)
    maxFit = np.amax(fitnessEjec)
    minFit = np.amin(fitnessEjec)
    avgFit = np.average(fitnessEjec)
    print('Mayor Fitness:', maxFit)
    print('Menor Fitness:', minFit)
    print('Promedio Fitness:', avgFit)
    print('###############################')

    cantEj = len(CurveArr)
    cmap = get_cmap(cantEj)

    fig, ax = plt.subplots()
    #ax.plot( Curve,color='dodgerblue', marker='o', markeredgecolor='dodgerblue', markerfacecolor='dodgerblue')
    for p in range(cantEj):
        ax.plot( CurveArr[p],color=cmap(p), marker='o', markeredgecolor=cmap(p), markerfacecolor=cmap(p))  
    ax.set_xlabel('Number of Iterations',fontsize=15)
    ax.set_ylabel('Fitness',fontsize=15)
    ax.set_title('African Vulture Optimization - ' + problem)
    plt.savefig(problem + '.jpg', format='jpg')
    #plt.show()