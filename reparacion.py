
import numpy as np
import factibilidad as fc 
import random

def reparaSimple(SCP, solution):
    indices = list(range(SCP["m"]))                                                     # genero una lista cuyos indices son la posición de las restricciones 
    random.shuffle(indices)                                                             # desordeno la lista para validar aleatoreamente la restriccion
    for i in indices:                                                                   # reviso cada una de las restricciones
        if np.sum(SCP["S"][i] * solution) < 1:                                          # si la restriccion es violada
            idxRestriccion = np.argwhere((SCP["S"][i]) > 0)                             # obtengo las columnas que satisfacen la restriccion
            idxMenorPeso = idxRestriccion[np.argmin(SCP["C_s"][idxRestriccion])]        # selecciono la que tenga menor costo
            solution[idxMenorPeso[0]] = 1                                               # coloco el u en el lugar que corresponde
        flag, aux = fc.check_sol(SCP, solution)                                         # valido la factibilidad de la solucion nuevamente
        if flag:    
            break                                                                       #solucion reparada, salgo del ciclo for
            
    return solution

def reparaComplejo(SCP, solution):
    sol =  np.reshape(solution, (SCP["n"],))
    set = SCP["S"]
    
    feasible, aux = fc.check_sol(SCP,sol)

    while not feasible:
        r_no_cubiertas = np.zeros((SCP["m"],))
        r_no_cubiertas[np.argwhere(aux == 0)] = 1           # Vector indica las restricciones no cubiertas
        cnc = np.dot(r_no_cubiertas, set)                   # Cantidad de restricciones no cubiertas que cubre cada columna (de tamaño n)
        trade_off = np.divide(SCP["C_s"],cnc)               # Trade off entre zonas no cubiertas y costo de seleccionar cada columna
        idx = np.argmin(trade_off)                          # Selecciono la columna con el trade off mas bajo
        sol[idx] = 1                                        # Asigno 1 a esa columna
        feasible, aux = fc.check_sol(SCP,sol)               # Verifico si la solucion actualizada es factible

    return sol