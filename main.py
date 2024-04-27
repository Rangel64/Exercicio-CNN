import cv2
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
import skimage.measure

entradas_train = pd.read_csv('all/archive/train/targets/train.csv')/255 * 2 -1 
targets_train = np.asarray(pd.read_csv('all/archive/train/targets/targetsTrain.csv'))

entradas_test = np.asarray(pd.read_csv('all/archive/test/targets/test.csv'))/255 * 2 -1 
targets_test = np.asarray(pd.read_csv('all/archive/test/targets/targetsTest.csv'))





class CNN:
    entrada = None
    target = None
    
    entrada_test = None
    target_test = None
    
    entradas = None
    saidas = None
    
    alpha = 0.01
    
    neuronios = 150
    
    kernels = [
        np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9,
        np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    ]
    
    camadaX = None
    camadaA = None
    camadaB = None
    camadaC = None
    camadaD = None
    camadaE = None
    camadaF = None
    camadaY = None
    
    A = None
    B = None
    C = None
    D = None
    E = None
    F = None
    Y = None
    
    XA = None
    AB = None
    BC = None
    CD = None
    DE = None
    EF = None
    FY = None
    
    
    biasA = None
    biasB = None
    biasC = None
    biasD = None
    biasE = None
    biasF = None
    biasY = None
    
    targetsTeste = []
    
    y_pred = []
    y_pred_test = []
    
    ciclo = 0
    
    erroTotal = np.Infinity
    erroTotal_test = np.Infinity
    trained = 0
    
    listaCiclo = []
    erros = []
    
    listaCiclo_test = []
    erros_test = []
    
    firstErro = 0
    
    erro_anterior = 0
     
    accuracy_train = 0
    accuracy_test = 0
    
    list_accuracy_train = []
    list_accuracy_test = []
    
    aleatorio = 0.5
    
    def convolution2D(self, image, kernel, stride=1, padding=0):
        # Calcula as dimensões da imagem de saída
        output_height = (image.shape[0] + 2 * padding - kernel.shape[0]) // stride + 1
        output_width = (image.shape[1] + 2 * padding - kernel.shape[1]) // stride + 1
        
        # Adiciona padding à imagem, se necessário
        if padding > 0:
            padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        else:
            padded_image = image
        
        # Inicializa a imagem de saída
        output = np.zeros_like(image)
        
        # Aplica a convolução
        for i in range(output_height):
            for j in range(output_width):
                # Calcula os índices da região na imagem original
                start_i = i * stride
                start_j = j * stride
                end_i = start_i + kernel.shape[0]
                end_j = start_j + kernel.shape[1]
                # Extrai a região da imagem correspondente ao kernel
                region = padded_image[start_i:end_i, start_j:end_j]
                # Realiza a multiplicação elemento a elemento entre a região e o kernel
                output[start_i, start_j] = np.sum(region * kernel)
        
        return output
    
    def max_pooling(self, image, pool_size=2, stride=2):
        # Tamanho da entrada
        input_height, input_width = image.shape
    
        # Tamanho da saída
        output_height = (input_height - pool_size) // stride + 1
        output_width = (input_width - pool_size) // stride + 1
    
        # Inicializa a matriz de saída
        output_image = np.zeros((output_height, output_width))
    
        # Realiza o MaxPooling
        for y in range(0, output_height):
            for x in range(0, output_width):
                # Define a janela a ser considerada
                window = image[y*stride:y*stride+pool_size, x*stride:x*stride+pool_size]
                # Calcula o máximo dentro da janela e atribui à matriz de saída
                output_image[y, x] = np.max(window)
    
        return output_image
    
    def flatten(self, image):
        return image.flatten()
    
    def get_weights_biases(self):
        return {
            'XA': self.XA,
            'AB': self.AB,
            'BC': self.BC,
            'CD': self.CD,
            'DE': self.DE,
            'EF': self.EF,
            'FY': self.FY,
            'biasA': self.biasA,
            'biasB': self.biasB,
            'biasC': self.biasC,
            'biasD': self.biasD,
            'biasE': self.biasE,
            'biasF': self.biasF,
            'biasY': self.biasY
        }
    
    def save_weights_biases(self, filename):
        
        weights_biases = self.get_weights_biases()
        
        with open(filename, 'wb') as file:
            pickle.dump(weights_biases, file)
    
    def load_weights_biases(self, filename):
        with open(filename, 'rb') as file:
            weights_biases = pickle.load(file)
            self.XA = weights_biases['XA']
            self.AB = weights_biases['AB']
            self.BC = weights_biases['BC']
            self.CD = weights_biases['CD']
            self.DE = weights_biases['DE']
            self.EF = weights_biases['EF']
            self.FY = weights_biases['FY']
            self.biasA = weights_biases['biasA']
            self.biasB = weights_biases['biasB']
            self.biasC = weights_biases['biasC']
            self.biasD = weights_biases['biasD']
            self.biasE = weights_biases['biasE']
            self.biasF = weights_biases['biasF']
            self.biasY = weights_biases['biasY']
            
    def relu(self,resultadosPuros):
        linhas,colunas = resultadosPuros.shape
        
        resultados = np.zeros((linhas,colunas))
        
        for i in range(linhas):
            for j in range(colunas):
                if(resultadosPuros[i][j] >= 0):
                    resultados[i,j] = resultadosPuros[i][j]
                else:
                    resultados[i,j] = 0
        
        return resultados
    
    def derivadaRelu(self,resultado):
        linhas, colunas = resultado.shape
        resultadoDerivada = np.zeros((linhas,colunas))
        
        for i in range(linhas):
            for j in range(colunas):
                if(resultado[i][j]>=0):
                    resultadoDerivada[i][j] = 1
                else:
                    resultadoDerivada[i][j] = 0
                    
        return resultadoDerivada
    
    def adaptive_learning_rate(self, erro):
        
        deltaErro = erro - self.erro_anterior
        
        if deltaErro < 0.5:
            self.alpha = self.alpha * 1.5  # Reduza a taxa de aprendizagem
        else:
            self.alpha = self.alpha * 0.5  # Aumente a taxa de aprendizagem

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def derivadaSigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def get_accuracy(self,entrada, target):

        entradas_conv = []
        
        for i in range(entrada.shape[0]):
            image_array = entrada[i]
            image = image_array.reshape(50, 50)
            image = self.conv2D(image, self.kernels)
            image = self.conv2D(image, self.kernels)
            entradas_conv.append(self.flatten(image))
        
        entradas_conv = np.vstack(entradas_conv)
        
        camadaA = np.dot(entradas_conv, self.XA) + self.biasA

        A = np.tanh(camadaA)
        
        camadaB = np.dot(A, self.AB) + self.biasB

        B = np.tanh(camadaB)
        
        camadaC = np.dot(B, self.BC) + self.biasC

        C = np.tanh(camadaC)
        
        camadaD = np.dot(C, self.CD) + self.biasD

        D = np.tanh(camadaD)
        
        camadaE = np.dot(D, self.DE) + self.biasE

        E = np.tanh(camadaE)
        
        camadaF = np.dot(E, self.EF) + self.biasF

        F = np.tanh(camadaF)
        
        camadaY = np.dot(F, self.FY) + self.biasY

        Y = np.tanh(camadaY)
        
        y_pred = np.where(Y > 0, 1, -1)
        
        accuracy = accuracy_score(target, y_pred)

        return accuracy
    
    def accuracy(self):
        accuracy_train = self.get_accuracy(self.entrada, self.target)
        accuracy_test = self.get_accuracy(self.entrada_test, self.target_test)

        return accuracy_train, accuracy_test
    
    def reset(self):
        print('================================')
        print('reinicando os pesos!!!')
        print('================================')
         
        for i in range(169):
            for j in range(self.neuronios):
                self.XA[i][j] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for j in range(self.neuronios):
            self.biasA[0][j] = rd.uniform(-self.aleatorio,self.aleatorio)
        
        for j in range(self.neuronios):
            for k in range(self.neuronios):
                self.AB[j][k] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for k in range(self.neuronios):
            self.biasB[0][k] = rd.uniform(-self.aleatorio,self.aleatorio)
            
        for k in range(self.neuronios):
            for l in range(self.neuronios):
                self.BC[k][l] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for l in range(self.neuronios):
            self.biasC[0][l] = rd.uniform(-self.aleatorio,self.aleatorio)
            
        for l in range(self.neuronios):
            for m in range(self.neuronios):
                self.CD[l][m] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for m in range(self.neuronios):
            self.biasD[0][m] = rd.uniform(-self.aleatorio,self.aleatorio)
            
        for m in range(self.neuronios):
            for n in range(self.neuronios):
                self.DE[m][n] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for n in range(self.neuronios):
            self.biasE[0][n] = rd.uniform(-self.aleatorio,self.aleatorio)
            
        for n in range(self.neuronios):
            for o in range(self.neuronios):
                self.EF[n][o] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for o in range(self.neuronios):
            self.biasF[0][o] = rd.uniform(-self.aleatorio,self.aleatorio)
            
        for o in range(self.neuronios):
            for p in range(self.saidas):
                self.FY[o][p] = rd.uniform(-self.aleatorio,self.aleatorio)
                   
        for p in range(self.saidas):
            self.biasY[0][p] = rd.uniform(-self.aleatorio,self.aleatorio)
               
        self.listaCiclo = []
        self.erros = []
        
        self.listaCiclo_test = []
        self.erros_test = []
        
        self.ciclo = 0
        
        self.erroTotal = np.Infinity
        
        self.trained = 0
        
        self.firstErro = 0
    
    def __init__(self, entrada,target,entradas_test,targets_test):
        self.entrada = entrada
        self.target = target
        self.entrada_test = entradas_test
        self.target_test = targets_test
        self.entradas = entrada.shape[1]
        self.saidas = target.shape[1]
        self.XA = np.zeros((169,self.neuronios))
        self.AB = np.zeros((self.neuronios,self.neuronios))
        self.BC = np.zeros((self.neuronios,self.neuronios))
        self.CD = np.zeros((self.neuronios,self.neuronios))
        self.DE = np.zeros((self.neuronios,self.neuronios))
        self.EF = np.zeros((self.neuronios,self.neuronios))
        self.FY = np.zeros((self.neuronios,self.saidas))
        self.biasA = np.zeros((1,self.neuronios))
        self.biasB = np.zeros((1,self.neuronios))
        self.biasC = np.zeros((1,self.neuronios))
        self.biasD = np.zeros((1,self.neuronios))
        self.biasE = np.zeros((1,self.neuronios))
        self.biasF = np.zeros((1,self.neuronios))
        self.biasY = np.zeros((1,self.saidas))
        self.camadaX = np.zeros((1,self.entradas))
        self.camadaA = np.zeros((1,self.neuronios))
        self.camadaB = np.zeros((1,self.neuronios))
        self.camadaC = np.zeros((1,self.neuronios))
        self.camadaD = np.zeros((1,self.neuronios))
        self.camadaE = np.zeros((1,self.neuronios))
        self.camadaF = np.zeros((1,self.neuronios))
        self.camadaY = np.zeros((1,self.saidas))
        self.listaCiclo = []
        self.erros = []
        self.listaCiclo_test = []
        self.erros_test = []
        self.firstErro = 0
    
    def conv2D(self, image, kernels):
        result = image
        for kernel in kernels:
            result = cv2.filter2D(result, -1, kernel)
        result = skimage.measure.block_reduce(result, (2,2), np.max)
        return result
    
    # def conv2D(self, image, kernels):
    #     result = image
    #     for kernel in kernels:
    #         result = self.convolution2D(result, kernel)
    #     result = skimage.measure.block_reduce(result, (2,2), np.max)
    #     print(result.shape)
    #     return result
    
    def train(self):
        
        if(self.trained==0):
            
            while self.ciclo<10000 and self.erroTotal>100 and self.accuracy_train<0.90:
                entradas_conv = []
                
                for i in range(self.entrada.shape[0]):
                    image_array = self.entrada.iloc[i].to_numpy()
                    image = image_array.reshape(50, 50)
                    image = self.conv2D(image, self.kernels)
                    image = self.conv2D(image, self.kernels)
                    entradas_conv.append(self.flatten(image))
                
                entradas_conv = np.vstack(entradas_conv)
                
                self.camadaA = np.dot(entradas_conv, self.XA) + self.biasA
    
                self.A = np.tanh(self.camadaA)
                
                self.camadaB = np.dot(self.A, self.AB) + self.biasB
    
                self.B = np.tanh(self.camadaB)
                
                self.camadaC = np.dot(self.B, self.BC) + self.biasC
    
                self.C = np.tanh(self.camadaC)
                
                self.camadaD = np.dot(self.C, self.CD) + self.biasD
    
                self.D = np.tanh(self.camadaD)
                
                self.camadaE = np.dot(self.D, self.DE) + self.biasE
    
                self.E = np.tanh(self.camadaE)
                
                self.camadaF = np.dot(self.E, self.EF) + self.biasF
    
                self.F = np.tanh(self.camadaF)
                
                self.camadaY = np.dot(self.F, self.FY) + self.biasY
    
                self.Y = np.tanh(self.camadaY)

                erro = np.sqrt(np.sum((self.target-self.Y)**2))
                
                erro_test, _ = self.test(entradas_test, targets_test)
                
                y_pred = np.where(self.Y > 0, 1, -1)
                
                self.accuracy_train = accuracy_score(self.target, y_pred)
                self.accuracy_test = self.get_accuracy(self.entrada_test, self.target_test)
                
                print('================================')
                print('Ciclo: ' + str(self.ciclo))
                print('Alpha: ' + str(self.alpha))
                print('Treinamento Acuracia: ' + str(self.accuracy_train))
                print('Treinamento LMSE: ' + str(erro))
                print('Teste Acuracia: ' + str(self.accuracy_test))
                print('Teste LMSE: ' + str(erro_test))
                print('================================')
                
                deltinhaFY = ((self.target-self.Y)/np.sqrt(np.sum((self.target-self.Y)**2)))*(1-self.Y**2)
                
                deltinhaEF = np.dot(deltinhaFY,self.FY.T)*(1-self.F**2)
                
                deltinhaDE = np.dot(deltinhaEF,self.EF.T)*(1-self.E**2)
                
                deltinhaCD = np.dot(deltinhaDE,self.DE.T)*(1-self.D**2)
                
                deltinhaBC = np.dot(deltinhaCD,self.CD.T)*(1-self.C**2)
                
                deltinhaAB = np.dot(deltinhaBC,self.BC.T)*(1-self.B**2)
                
                deltinhaXA = np.dot(deltinhaAB,self.AB.T)*(1-self.A**2)
                
                deltaFY = self.alpha * np.dot(deltinhaFY.transpose(),self.F)
                deltaBiasY = self.alpha * np.sum(deltinhaFY)
                    
                deltaEF = self.alpha * np.dot(deltinhaEF.transpose(),self.E)
                deltaBiasF = self.alpha * np.sum(deltinhaEF)
                    
                deltaDE = self.alpha * np.dot(deltinhaDE.transpose(),self.D)
                deltaBiasE = self.alpha * np.sum(deltinhaDE)
                    
                deltaCD = self.alpha * np.dot(deltinhaCD.transpose(),self.C)
                deltaBiasD = self.alpha * np.sum(deltinhaCD)  
                
                deltaBC = self.alpha * np.dot(deltinhaBC.transpose(),self.B)
                deltaBiasC = self.alpha * np.sum(deltinhaBC) 
                
                deltaAB = self.alpha * np.dot(deltinhaAB.transpose(),self.A)
                deltaBiasB = self.alpha * np.sum(deltinhaAB) 
                
                deltaXA = self.alpha * np.dot(deltinhaXA.transpose(),entradas_conv)
                deltaBiasA = self.alpha * np.sum(deltinhaXA)
                
                self.FY = self.FY + deltaFY.transpose()
                self.biasY = self.biasY + deltaBiasY.transpose()
                
                self.EF = self.EF + deltaEF.transpose()
                self.biasF = self.biasF + deltaBiasF.transpose()
 
                self.DE = self.DE + deltaDE.transpose()
                self.biasE = self.biasE + deltaBiasE.transpose()
                
                self.CD = self.CD + deltaCD.transpose()
                self.biasD = self.biasD + deltaBiasD.transpose()
                
                self.BC = self.BC + deltaBC.transpose()
                self.biasC = self.biasC + deltaBiasC.transpose()
                
                self.AB = self.AB + deltaAB.transpose()
                self.biasB = self.biasB + deltaBiasB.transpose()
                
                self.XA = self.XA + deltaXA.transpose()
                self.biasA = self.biasA + deltaBiasA.transpose()
                    
                self.list_accuracy_train.append(self.accuracy_train)
                self.list_accuracy_test.append(self.accuracy_test)
                  
                self.listaCiclo.append(self.ciclo)

                self.erros.append(erro)
                    
                self.erros_test.append(erro_test)
                             
                data = {
                    'Epoch': self.listaCiclo,
                    'Error': self.erros,
                    'Test Error': self.erros_test,
                    'Accuracy': self.list_accuracy_train,
                    'Test Accuracy': self.list_accuracy_test
                }
                
                df = pd.DataFrame(data)

                # Plotar os gráficos
                plt.figure(figsize=(12, 6))
            
                plt.subplot(1, 2, 1)
                sns.lineplot(x='Epoch', y='Error', data=df, label='Training Error')
                sns.lineplot(x='Epoch', y='Test Error', data=df, label='Test Error')
                plt.xlabel('Epoch')
                plt.ylabel('Error')
                plt.title('Training and Test Error over Epochs')
            
                plt.subplot(1, 2, 2)
                sns.lineplot(x='Epoch', y='Accuracy', data=df, label='Training Accuracy')
                sns.lineplot(x='Epoch', y='Test Accuracy', data=df, label='Test Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Training and Test Accuracy over Epochs')
            
                plt.tight_layout()
                plt.show()
                
                self.ciclo = self.ciclo + 1
                
            self.trained = 1
        
        else:
            print('================================')
            print('Modelo ja treinado')
            print('================================')   
        
    
    def test(self, entrada,target):
        
        entradas_conv = []
        
        for i in range(entrada.shape[0]):
            image_array = entrada[i]
            image = image_array.reshape(50, 50)
            image = self.conv2D(image, self.kernels)
            image = self.conv2D(image, self.kernels)
            entradas_conv.append(self.flatten(image))
        
        entradas_conv = np.vstack(entradas_conv)
        
        camadaA = np.dot(entradas_conv, self.XA) + self.biasA

        A = np.tanh(camadaA)
        
        camadaB = np.dot(A, self.AB) + self.biasB

        B = np.tanh(camadaB)
        
        camadaC = np.dot(B, self.BC) + self.biasC

        C = np.tanh(camadaC)
        
        camadaD = np.dot(C, self.CD) + self.biasD

        D = np.tanh(camadaD)
        
        camadaE = np.dot(D, self.DE) + self.biasE

        E = np.tanh(camadaE)
        
        camadaF = np.dot(E, self.EF) + self.biasF

        F = np.tanh(camadaF)
        
        camadaY = np.dot(F, self.FY) + self.biasY

        Y = np.tanh(camadaY)
            
        erro = np.sqrt(np.sum((target-Y)**2))
        
        Y = np.where(Y > 0, 1, -1)
             
        return erro,Y

                 
model = CNN(entradas_train, targets_train, entradas_test, targets_test)
model.reset()
model.train()


# model.save_weights_biases('pesosMLP/pesos.pkl')



# model2 = Perceptron(entradas_train, targets_train, entradas_test, targets_test)

# model2.load_weights_biases('pesosMLP/pesos.pkl')

# precisao_train = model2.get_accuracy(entradas_train, targets_train)
# erro_train, _ = model2.test(entradas_train, targets_train)

# print('==========================================================')
# print('Acuracia Treinamento: ' + str(precisao_train))
# print('Erro Treinamento: ' + str(erro_train))
# print('==========================================================')

# precisao_test = model2.get_accuracy(entradas_test, targets_test)
# erro_test, _ = model2.test(entradas_test, targets_test)

# print('==========================================================')
# print('Acuracia Teste: ' + str(precisao_test))
# print('Erro Teste: ' + str(erro_test))
# print('==========================================================')



