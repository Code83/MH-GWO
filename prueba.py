
import numpy as np
import leerProblema as Instance
import factibilidad as fc
import fitness as fit
import reparacion as rep
    


dirSCP = 'SCP/scp41.txt'

SCP = Instance.read_instance(dirSCP)

solucion = np.random.randint(low=0, high=2, size = (SCP["n"]))

print(solucion.tolist())

flag, aux = fc.check_sol(SCP,solucion)
print("¿Solucion factible?: "+str(flag))
print("Coberturas de la solucion: "+str(aux.tolist()))

if flag:
    # si tengo una solucion factible calculo el fitness
    fitness = fit.fitness(SCP, solucion)
    print("Fitness de la solucion: "+str(fitness))
else:
    # si no tengo una solucion factible, reparo y luego calculo el fitness
    # solucion = rep.reparaSimple(SCP, solucion)
    solucion = rep.reparaComplejo(SCP, solucion)
    fitness = fit.fitness(SCP, solucion)
    print("Fitness de la solucion: "+str(fitness))
