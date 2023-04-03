import numpy as np

def check_sol(SCP, sol):
    flag = True
    set = SCP["S"]
    aux = np.dot(set,sol)           # Multiplico la matriz de cobertura por la solucion

    if 0 in aux:                    # Si alguna zona no esta cubierta, retorno False
        flag = False

    return flag, aux