import numpy as np
from scipy import special as scyesp

"""
Realiza una transformación de una matrix en el dominio de los continuos a una matriz binaria.
Usada para discretización de soluciones en metaheurísticas. 
Donde cada fila es una solución (individuo de la población), y las columnas de la matriz corresponden a las dimensiones de la solución.
Params
----------
matrixCont : matriz de continuos.
matrixBin  : matriz de poblacion binarizada en t-1. 
bestRowBin : individuo (fila) de mejor fitness.
SolutionRanking: Lista de indices ordenadas por fitness, en la posición 0 esta el best
transferFunction: Funciones de transferencia (V1,..,V4, S1,..,S4)
binarizationOperator: Operador de binarización (Standard,Complement, Elitist, Static, Roulette)
Returns
-------
matrixBinOut: matriz binaria.

Definiciones
---------
Para t=0, es decir, si matrixBin es vacía, se utiliza por defecto binarizationOperator = Standard.
"""

class DiscretizationScheme:
    
    def __init__(self, matrixCont, matrixBin, SolutionRanking, transferFunction, binarizationOperator):

        self.transferFunction = transferFunction
        self.binarizationOperator = binarizationOperator

        self.matrixCont = matrixCont
        self.matrixBin = matrixBin
        self.SolutionRanking = SolutionRanking
        self.bestRow = np.argmin(SolutionRanking) 

        #output
        self.matrixProbT = np.zeros(self.matrixCont.shape)
        self.matrixBinOut = np.zeros(self.matrixBin.shape)

        # debug
        self.matrixProbTAux = np.zeros(self.matrixCont.shape)
        self.matrixBinOutAux = np.zeros(self.matrixBin.shape)

        #Constantes O1
        self.o1a= 0
        self.o1b= 1
        self.o1c= 1
        self.o1d= 0
        
        #Constante de Q-Shaped
        self.xMax = np.max(0.5*self.matrixCont)

        self.uAlpha = 0.5 #*** Por definir como ingresar
        self.uBeta = 1.5 #*** Por definir como ingresar


    #Funciones de Transferencia
    def T_V1(self):
        self.matrixProbT = np.abs(scyesp.erf(np.divide(np.sqrt(np.pi),2)*self.matrixCont))

    def T_V2(self):
        self.matrixProbT = np.abs(np.tanh(self.matrixCont))

    def T_V3(self):
        self.matrixProbT = np.abs(np.divide(self.matrixCont, np.sqrt(1+np.power(self.matrixCont,2))))

    def T_V4(self):
        self.matrixProbT = np.abs( np.divide(2,np.pi)*np.arctan( np.divide(np.pi,2)*self.matrixCont ))

    def T_S1(self):
        self.matrixProbT = np.divide(1, ( 1 + np.exp(-2*self.matrixCont) ) )

    def T_S2(self):
        self.matrixProbT = np.divide(1, ( 1 + np.exp(-1*self.matrixCont) ) )

    def T_S3(self):
        self.matrixProbT = np.divide(1, ( 1 + np.exp(np.divide(-1*self.matrixCont,2) ) ))

    def T_S4(self):
        self.matrixProbT = np.divide(1, ( 1 + np.exp(np.divide(-1*self.matrixCont,3) ) ))

    def T_O1(self):
        self.matrixProbT = np.sin(np.multiply(np.multiply(np.multiply(np.multiply(2,np.pi),(self.matrixCont - self.o1a)),self.o1b),np.cos(np.multiply(np.multiply(np.multiply(2,np.pi),(self.matrixCont - self.o1a)),self.o1c))))+ self.o1d

    def T_O2(self): #*** solamente entregaría 1 o 0
        self.matrixProbT =np.trunc(np.abs(np.mod(self.matrixCont,2)))

    def T_O3(self):
        self.matrixProbT = np.divide((self.matrixCont + np.min(self.matrixCont)),(np.abs(np.min(self.matrixCont))+np.max(self.matrixCont)))

    def T_O4(self): #*** está fuera del dominio [0,1]
        self.matrixProbT = self.matrixCont

    def T_Q1(self): 
        self.matrixProbT[self.matrixCont < self.xMax] = np.abs(np.divide(self.matrixCont[self.matrixCont < self.xMax],np.max(0.5*self.matrixCont[self.matrixCont < self.xMax])))
        self.matrixProbT[self.matrixCont >= self.xMax] = 1

    def T_Q2(self):
        self.matrixProbT[self.matrixCont < self.xMax] = np.power(np.divide(self.matrixCont[self.matrixCont < self.xMax],np.max(0.5*self.matrixCont[self.matrixCont < self.xMax])),2)
        self.matrixProbT[self.matrixCont >= self.xMax] = 1

    def T_Q3(self): 
        self.matrixProbT[self.matrixCont < self.xMax] = np.power(np.divide(self.matrixCont[self.matrixCont < self.xMax],np.max(0.5*self.matrixCont[self.matrixCont < self.xMax])),3)
        self.matrixProbT[self.matrixCont >= self.xMax] = 1

    def T_Q4(self): 
        self.matrixProbT[self.matrixCont < self.xMax] = np.power(np.divide(self.matrixCont[self.matrixCont < self.xMax],np.max(0.5*self.matrixCont[self.matrixCont < self.xMax])),4)
        self.matrixProbT[self.matrixCont >= self.xMax] = 1

    # def T_U(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
    #     self.matrixProbT = self.uAlpha*(np.power(np.abs(self.matrixCont),self.uBeta))

    def T_U1(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 0.5*(np.power(np.abs(self.matrixCont),1.5))

    def T_U2(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 0.5*(np.power(np.abs(self.matrixCont),2.75))

    def T_U3(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 0.5*(np.power(np.abs(self.matrixCont),4))

    def T_U4(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 1.25*(np.power(np.abs(self.matrixCont),1.5))

    def T_U5(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 1.25*(np.power(np.abs(self.matrixCont),2.75))

    def T_U6(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 1.25*(np.power(np.abs(self.matrixCont),4))

    def T_U7(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 2*(np.power(np.abs(self.matrixCont),1.5))

    def T_U8(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 2*(np.power(np.abs(self.matrixCont),2.75))

    def T_U9(self): #Hay que ver como definir como ingresar self.uAlpha y self.uBeta
        self.matrixProbT = 2*(np.power(np.abs(self.matrixCont),4))

    def T_Z1(self): 
        self.matrixProbT = np.power((1-np.power(2,self.matrixCont)),0.5)

    def T_Z2(self): 
        self.matrixProbT = np.power((1-np.power(5,self.matrixCont)),0.5)

    def T_Z3(self): 
        self.matrixProbT = np.power((1-np.power(8,self.matrixCont)),0.5)

    def T_Z4(self): 
        self.matrixProbT = np.power((1-np.power(20,self.matrixCont)),0.5)

    def T_X1(self): #funciones de transferencias S1 invertidad
        self.matrixProbT = np.divide(1, ( 1 + np.exp(2*self.matrixCont) ) )

    def T_X2(self): #funciones de transferencias S2 invertidad
        self.matrixProbT = np.divide(1, ( 1 + np.exp(1*self.matrixCont) ) )

    def T_X3(self): #funciones de transferencias S3 invertidad
        self.matrixProbT = np.divide(1, ( 1 + np.exp(np.divide(1*self.matrixCont,2) ) ))

    def T_X4(self): #funciones de transferencias S4 invertidad
        self.matrixProbT = np.divide(1, ( 1 + np.exp(np.divide(1*self.matrixCont,3) ) ))

    #Binarization
    def B_Standard(self):
        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixCont.shape)
        self.matrixBinOut = np.greater(self.matrixProbT,matrixRand).astype(int)

    def B_Complement(self):
        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixCont.shape)
        matrixComplement = np.abs(1-self.matrixBin)
        self.matrixBinOut = np.multiply(np.greater_equal(self.matrixProbT,matrixRand).astype(int),matrixComplement)

    def B_Elitist(self):
        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixCont.shape)
        # greater, porque es estricto en la ecuacion.
        conditionMatrix = np.greater(self.matrixProbT,matrixRand)
        #todo: validar que el index exista
        bestIndividual = self.matrixBin[self.bestRow]
        # si ProbT > Rand() , then bestIndividualBin, else 0
        self.matrixBinOut = np.where(conditionMatrix==True,bestIndividual,0)

    def B_Static(self):
        alfa = 1/3
        self.matrixBinOut[self.matrixProbT<=alfa] = 0
        self.matrixBinOut[(self.matrixProbT > alfa) & (self.matrixProbT <= 0.5*(1+alfa))] = self.matrixBin[(self.matrixProbT > alfa) & (self.matrixProbT <= 0.5*(1+alfa))]
        self.matrixBinOut[self.matrixProbT>=0.5*(1+alfa)] = 1

    def B_ElitistRoulette(self):
        matrixRand = np.random.uniform(low=0.0,high=1.0,size=self.matrixCont.shape)
        #greater, porque es estricto en la ecuacion.
        conditionMatrix = np.greater(self.matrixProbT,matrixRand)
        #todo: validar que el index exista
        alfa = 0.2
        # condicion sum()==0, para el caso en que entregamos lista de rank con [0 0 0 0 0 0 0 0 ... 0 0 0 0] -> cuando generamos poblacion inicial
        if (self.SolutionRanking.shape[0]*alfa < 1) or (self.SolutionRanking.sum()==0):
            bestIndividual = self.matrixBin[0]
        else:
            BestSolutionRaking = int(self.SolutionRanking.shape[0] * alfa)
            random = np.random.randint(low = 0, high = BestSolutionRaking)
            bestIndividual = self.matrixBin[random]
        self.matrixBinOut = np.where(conditionMatrix==True,bestIndividual,0)

    def binariza(self):
        if self.transferFunction == 'V1':
            self.T_V1()

        if self.transferFunction == 'V2':
            self.T_V2()

        if self.transferFunction == 'V3':
            self.T_V3()

        if self.transferFunction == 'V4':
            self.T_V4()

        if self.transferFunction == 'S1':
            self.T_S1()

        if self.transferFunction == 'S2':
            self.T_S2()

        if self.transferFunction == 'S3':
            self.T_S3()

        if self.transferFunction == 'S4':
            self.T_S4()

        if self.transferFunction == 'O1':
            self.T_O1()

        if self.transferFunction == 'Z1':
            self.T_Z1()

        if self.transferFunction == 'Z2':
            self.T_Z2()

        if self.transferFunction == 'Z3':
            self.T_Z3()

        if self.transferFunction == 'Z4':
            self.T_Z4()

        if self.transferFunction == 'X1':
            self.T_X1()

        if self.transferFunction == 'X2':
            self.T_X2()

        if self.transferFunction == 'X3':
            self.T_X3()

        if self.transferFunction == 'X4':
            self.T_X4()

        if self.binarizationOperator == 'Standard':
            self.B_Standard()

        if self.binarizationOperator == 'Complement':
            self.B_Complement()

        if self.binarizationOperator == 'Elitist':
            self.B_Elitist()

        if self.binarizationOperator == 'Static':
            self.B_Static()

        if self.binarizationOperator == 'ElitistRoulette':
            self.B_ElitistRoulette()

        return self.matrixBinOut

    def appliedTransferFunction(self):
        if self.transferFunction == 'V1':
            self.T_V1()

        if self.transferFunction == 'V2':
            self.T_V2()

        if self.transferFunction == 'V3':
            self.T_V3()

        if self.transferFunction == 'V4':
            self.T_V4()

        if self.transferFunction == 'S1':
            self.T_S1()

        if self.transferFunction == 'S2':
            self.T_S2()

        if self.transferFunction == 'S3':
            self.T_S3()

        if self.transferFunction == 'S4':
            self.T_S4()

        if self.transferFunction == 'O1':
            self.T_O1()

        if self.transferFunction == 'O2':
            self.T_O2()

        if self.transferFunction == 'O3':
            self.T_O3()

        if self.transferFunction == 'O4':
            self.T_O4()

        if self.transferFunction == 'Q1':
            self.T_Q1()

        if self.transferFunction == 'Q2':
            self.T_Q2()

        if self.transferFunction == 'Q3':
            self.T_Q3()

        if self.transferFunction == 'Q4':
            self.T_Q4()

        if self.transferFunction == 'Z1':
            self.T_Z1()

        if self.transferFunction == 'Z2':
            self.T_Z2()

        if self.transferFunction == 'Z3':
            self.T_Z3()

        if self.transferFunction == 'Z4':
            self.T_Z4()

        if self.transferFunction == 'X1':
            self.T_X1()

        if self.transferFunction == 'X2':
            self.T_X2()

        if self.transferFunction == 'X3':
            self.T_X3()

        if self.transferFunction == 'X4':
            self.T_X4()

        return self.matrixProbT
