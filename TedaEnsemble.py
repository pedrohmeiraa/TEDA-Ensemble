from padasip.filters.base_filter import AdaptiveFilter

import pandas as pd
import numpy as np
import padasip as pa

np.random.seed(0)

class DataCloud:
  N=0
  def __init__(self, nf, mu, w, x):
      self.n=1
      self.nf=nf
      self.mean=x
      self.mu = mu
      self.w = w
      self.variance=0
      self.pertinency=1
      self.faux = pa.filters.FilterRLS(n=self.nf, mu=self.mu, w=self.w)
      DataCloud.N+=1
  def addDataCloud(self,x):
      self.n=2
      self.mean=(self.mean+x)/2
      self.variance=((np.linalg.norm(self.mean-x))**2)
  def updateDataCloud(self,n,mean,variance, faux):
      self.n=n
      self.mean=mean
      self.variance=variance
      self.faux = faux
  
 
class TEDAEnsemble:
  c= np.array([DataCloud(nf=2, mu=0.9, w=[0,0], x=0)],dtype=DataCloud)
  alfa= np.array([0.0],dtype=float)
  intersection = np.zeros((1,1),dtype=int)
  listIntersection = np.zeros((1),dtype=int)
  matrixIntersection = np.zeros((1,1),dtype=int)
  relevanceList = np.zeros((1),dtype=int)
  k=1

  def __init__(self, m, mu, activation_function, threshold):
    TEDAEnsemble.m = m
    TEDAEnsemble.mu = mu
    TEDAEnsemble.activation_function = activation_function
    TEDAEnsemble.threshold = threshold


    TEDAEnsemble.alfa= np.array([0.0],dtype=float)
    TEDAEnsemble.intersection = np.zeros((1,1),dtype=int)
    TEDAEnsemble.listIntersection = np.zeros((1),dtype=int)
    TEDAEnsemble.relevanceList = np.zeros((1),dtype=int)
    TEDAEnsemble.matrixIntersection = np.zeros((1,1),dtype=int)
    TEDAEnsemble.k=1
    TEDAEnsemble.classIndex = [[1.0],[1.0]] #<========== try in another moment: [np.array(1.0),np.array(1.0)]
    TEDAEnsemble.argMax = []
    TEDAEnsemble.RLSF_Index = []
    TEDAEnsemble.Ypred = []
    TEDAEnsemble.Ypred_STACK = []
    TEDAEnsemble.Ypred_POND = []
    TEDAEnsemble.Ypred_MAJOR = []
    TEDAEnsemble.X_ant = np.zeros((1, TEDAEnsemble.m), dtype=float)
    TEDAEnsemble.NumberOfFilters = []
    TEDAEnsemble.NumberOfDataClouds = []
    
    np.random.seed(0)
    TEDAEnsemble.random_factor = np.random.rand(TEDAEnsemble.m-1, TEDAEnsemble.m)
    #TEDAEnsemble.random_factor = np.array([[0.00504779, 0.99709118]])

    if (TEDAEnsemble.activation_function == "He"):
      factor = np.sqrt(2/(TEDAEnsemble.m-1))
    elif (TEDAEnsemble.activation_function == "Xavier"):
      factor = np.sqrt(1/(TEDAEnsemble.m-1))
    elif (TEDAEnsemble.activation_function == "Yoshua"):
      factor = np.sqrt(2/(TEDAEnsemble.m+(TEDAEnsemble.m-1)))
    elif (TEDAEnsemble.activation_function == "zero"):
      factor = 0
    else: #Utiliza a Formula do "He" como DEFAULT
      factor = np.sqrt(2/TEDAEnsemble.m-1)
           
    TEDAEnsemble.w_init = TEDAEnsemble.random_factor*factor 
    TEDAEnsemble.w_init = TEDAEnsemble.w_init[0].tolist()
    #print("w_init do TEDA: ", TEDAEnsemble.w_init)
    TEDAEnsemble.c = np.array([DataCloud(nf=TEDAEnsemble.m, mu=TEDAEnsemble.mu, w=TEDAEnsemble.w_init, x=0)],dtype=DataCloud)
    TEDAEnsemble.f0 = pa.filters.FilterRLS(TEDAEnsemble.m, mu=TEDAEnsemble.mu, w=TEDAEnsemble.w_init)


  def mergeClouds(self):
    i=0
    while(i<len(TEDAEnsemble.listIntersection)-1):
      #print("i do merge",i)
      #print(TEDAEnsemble.listIntersection)
      #print(TEDAEnsemble.matrixIntersection)
      merge = False
      j=i+1
      while(j<len(TEDAEnsemble.listIntersection)):
        #print("j do merge",j)
        #print("i",i,"j",j,"l",np.size(TEDAEnsemble.listIntersection),"m",np.size(TEDAEnsemble.matrixIntersection),"c",np.size(TEDAEnsemble.c))
        if(TEDAEnsemble.listIntersection[i] == 1 and TEDAEnsemble.listIntersection[j] == 1):
          TEDAEnsemble.matrixIntersection[i,j] = TEDAEnsemble.matrixIntersection[i,j] + 1;
          #print(TEDAEnsemble.matrixIntersection)
        nI = TEDAEnsemble.c[i].n
        nJ = TEDAEnsemble.c[j].n
        #print("I: ",list(TEDAEnsemble.c).index(TEDAEnsemble.c[i]), ". J: ",list(TEDAEnsemble.c).index(TEDAEnsemble.c[j]))
        meanI = TEDAEnsemble.c[i].mean
        meanJ = TEDAEnsemble.c[j].mean
        varianceI = TEDAEnsemble.c[i].variance
        varianceJ = TEDAEnsemble.c[j].variance
        nIntersc = TEDAEnsemble.matrixIntersection[i,j]
        fauxI = TEDAEnsemble.c[i].faux    #fauxI = TEDAEnsemble.RLS_Filters[i]
        fauxJ = TEDAEnsemble.c[j].faux    #fauxJ = TEDAEnsemble.RLS_Filters[j]
        
        wI = fauxI.getW()
        wJ = fauxJ.getW()
        #print("wJ: ", wJ)
        
        dwI = fauxI.getdW()
        dwJ = fauxJ.getdW()
                
        W = (nI*wI)/(nI + nJ) + (nJ*wJ)/(nI + nJ)

        if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
          #print(nIntersc, "(nIntersc) >", nI, "-", nIntersc, "=", nI - nIntersc, "(nI - nIntersc) OR", nIntersc, "(nIntersc) >", nJ, "-", nIntersc, "=", nJ - nIntersc, "(nJ - nIntersc)")
          
          #print("(nIntersc) =",nIntersc, ",nI =", nI, ",nJ =", nJ)
          #print("Juntou!")

          merge = True
          #update values
          n = nI + nJ - nIntersc
          mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
          variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
          #faux = fauxI #Considerando o de maior experiencia (mais amostras)
          #faux = pa.filters.FilterRLS(2, mu=mu_, w = W) #Considerando a inicialização dos pesos (W)
                       
          if (nI >= nJ):
            fauxI.mergeWith(fauxJ, nI, nJ)
            faux = fauxI

          else:
            fauxJ.mergeWith(fauxI, nI, nJ)
            faux = fauxJ
          
          #TEDAEnsemble.RLS_Filters.pop()

          newCloud = DataCloud(nf=TEDAEnsemble.m, mu=TEDAEnsemble.mu, w=TEDAEnsemble.w_init, x=mean)
          newCloud.updateDataCloud(n,mean,variance, faux)
          
          #atualizando lista de interseção
          TEDAEnsemble.listIntersection = np.concatenate((TEDAEnsemble.listIntersection[0 : i], np.array([1]), TEDAEnsemble.listIntersection[i + 1 : j],TEDAEnsemble.listIntersection[j + 1 : np.size(TEDAEnsemble.listIntersection)]),axis=None)
          #print("listInters dps de att:", TEDAEnsemble.listIntersection)
          #atualizando lista de data clouds
          #print("dentro do if do merge antes", TEDAEnsemble.c)
          TEDAEnsemble.c = np.concatenate((TEDAEnsemble.c[0 : i ], np.array([newCloud]), TEDAEnsemble.c[i + 1 : j],TEDAEnsemble.c[j + 1 : np.size(TEDAEnsemble.c)]),axis=None)
          #print("dentro do if do merge dps do concate", TEDAEnsemble.c)

          #update  intersection matrix
          M0 = TEDAEnsemble.matrixIntersection
          #Remover linhas 
          M1=np.concatenate((M0[0 : i , :],np.zeros((1,len(M0))),M0[i + 1 : j, :],M0[j + 1 : len(M0), :]))
          #remover colunas
          M1=np.concatenate((M1[:, 0 : i ],np.zeros((len(M1),1)),M1[:, i+1 : j],M1[:, j+1 : len(M0)]),axis=1)
          #calculando nova coluna
          col = (M0[:, i] + M0[:, j])*(M0[: , i]*M0[:, j] != 0)
          col = np.concatenate((col[0 : j], col[j + 1 : np.size(col)]))
          #calculando nova linha
          lin = (M0[i, :]+M0[j, :])*(M0[i, :]*M0[j, :] != 0)
          lin = np.concatenate((lin[ 0 : j], lin[j + 1 : np.size(lin)]))
          #atualizando coluna
          M1[:,i]=col
          #atualizando linha
          M1[i,:]=lin
          M1[i, i + 1 : j] = M0[i, i + 1 : j] + M0[i + 1 : j, j].T;   
          TEDAEnsemble.matrixIntersection = M1
          #print(TEDAEnsemble.matrixIntersection)
        j += 1
      if (merge):
        i = 0
      else:
        i += 1
				
  def run(self,X):
    TEDAEnsemble.listIntersection = np.zeros((np.size(TEDAEnsemble.c)),dtype=int)
    #print("k=", TEDAEnsemble.k)
    if TEDAEnsemble.k==1:
      TEDAEnsemble.c[0]=DataCloud(nf=TEDAEnsemble.m, mu=TEDAEnsemble.mu, w=TEDAEnsemble.w_init, x=X)
      TEDAEnsemble.argMax.append(0)
      TEDAEnsemble.c[0].faux = TEDAEnsemble.f0 #TEDAEnsemble.RLS_Filters = [TEDAEnsemble.f0] 
      TEDAEnsemble.RLSF_Index.append(0)
      TEDAEnsemble.X_ant = X
      #TEDAEnsemble.c[0].faux.adapt(X[1], TEDAEnsemble.X_ant)

    elif TEDAEnsemble.k==2:
      TEDAEnsemble.c[0].addDataCloud(X)
      TEDAEnsemble.argMax.append(0)
      TEDAEnsemble.RLSF_Index.append(0)
      TEDAEnsemble.X_ant = X
      #TEDAEnsemble.c[0].faux.adapt(X[1], TEDAEnsemble.X_ant)
    
    elif TEDAEnsemble.k>=3:
      i=0
      createCloud = True
      TEDAEnsemble.alfa = np.zeros((np.size(TEDAEnsemble.c)),dtype=float)

      for data in TEDAEnsemble.c:
        n= data.n + 1
        mean = ((n-1)/n)*data.mean + (1/n)*X
        variance = ((n-1)/n)*data.variance +(1/n)*((np.linalg.norm(X-mean))**2)
        eccentricity=(1/n)+((mean-X).T.dot(mean-X))/(n*variance)
        typicality = 1 - eccentricity
        norm_eccentricity = eccentricity/2
        norm_typicality = typicality/(TEDAEnsemble.k-2)
        faux_ = data.faux
        #faux_.adapt(X[-1], TEDAEnsemble.X_ant)
        
        if (norm_eccentricity<=(TEDAEnsemble.threshold**2 +1)/(2*n)): #Se couber dentro da Cloud
          
          data.updateDataCloud(n,mean,variance, faux_)
          TEDAEnsemble.alfa[i] = norm_typicality
          createCloud= False
          TEDAEnsemble.listIntersection.itemset(i,1)
          #print("pesos=", data.faux.w)
          #print("x_ant=", TEDAEnsemble.X_ant)
          #print("dentro da cloud")
          faux_.adapt(X[-1], TEDAEnsemble.X_ant)
        #TEDAEnsemble.c[i].faux.adapt(X[-1], TEDAEnsemble.X_ant) #data.faux.adapt(X[-1], TEDAEnsemble.X_ant) #data.faux.adapt(X[-1], TEDAEnsemble.X_ant)
              
        else: #Se nao couber
          TEDAEnsemble.alfa[i] = norm_typicality
          TEDAEnsemble.listIntersection.itemset(i,0)
          #print("fora da cloud")
          #print("fora -> i:", i, " - Filtro: ", faux_)
          #faux_.adapt(X[-1], TEDAEnsemble.X_ant)
          #TEDAEnsemble.c[i].faux.adapt(X[-1], TEDAEnsemble.X_ant) #data.faux.adapt(X[-1], TEDAEnsemble.X_ant)
        i+=1

      if (createCloud):
        #print("no if de criar TEDAEnsemble:", TEDAEnsemble.c)
        TEDAEnsemble.c = np.append(TEDAEnsemble.c,DataCloud(nf=TEDAEnsemble.m, mu=TEDAEnsemble.mu, w=TEDAEnsemble.w_init, x=X))
        #print("dps do if TEDAEnsemble:", TEDAEnsemble.c)
        TEDAEnsemble.listIntersection = np.insert(TEDAEnsemble.listIntersection,i,1)
        TEDAEnsemble.matrixIntersection = np.pad(TEDAEnsemble.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0))
        #print("DataCloud Created!")
        #TEDAEnsemble.RLS_Filters.append(pa.filters.FilterRLS(TEDAEnsemble.m, mu=TEDAEnsemble.mu, w=TEDAEnsemble.w_init))


      #print("TEDAEnsemble antes do Merge:", TEDAEnsemble.c)
      
      #print("TEDAEnsemble dps do Merge:", TEDAEnsemble.c)
      TEDAEnsemble.NumberOfFilters.append(len(TEDAEnsemble.c))
      TEDAEnsemble.relevanceList = TEDAEnsemble.alfa /np.sum(TEDAEnsemble.alfa)
      TEDAEnsemble.argMax.append(np.argmax(TEDAEnsemble.relevanceList))
      TEDAEnsemble.classIndex.append(TEDAEnsemble.alfa)
      #print("Alfa", TEDAEnsemble.alfa)
      #print("argmax", np.argmax(TEDAEnsemble.relevanceList))
      #print("relevance list: ", TEDAEnsemble.relevanceList)
          
      filtro_usado = TEDAEnsemble.c[np.argmax(TEDAEnsemble.relevanceList)].faux

      #print("filtro_usado:", filtro_usado)
      #print("_______")
      
      self.mergeClouds()
    
    
      F_used = []
      N_used = []
      YX = []
      NX = []
    
      c=0

      for x in TEDAEnsemble.c:
        c = c+1
        fx = x.faux
        Nx = x.n
  
        F_used.append(fx)
        N_used.append(Nx)
      
      for i in range(0, len(F_used)): 
        
        yx_pred = F_used[i].predict(X)
        nx_pred = N_used[i]
        YX.append(yx_pred)
        NX.append(nx_pred)
        
      #Stacking
      y_pred_stack = sum(YX)/len(YX)
      #print("ypred_bag: ", y_pred_bag)
        
      #Pondering     
      y_pred_pond = np.sum(np.multiply(YX, NX))/np.sum(NX)
      #print("ypred_pond", y_pred_pond)
      
      #Best of all
      y_pred_major = filtro_usado.predict(X)
      #print("ypred_major", y_pred_major)        
      #print("___________")

      TEDAEnsemble.Ypred.append(y_pred_pond)
      TEDAEnsemble.Ypred_STACK.append(y_pred_stack)
      TEDAEnsemble.Ypred_POND.append(y_pred_pond)
      TEDAEnsemble.Ypred_MAJOR.append(y_pred_major)
        
      TEDAEnsemble.RLSF_Index.append(np.argmax(TEDAEnsemble.relevanceList))
      TEDAEnsemble.X_ant = X
      
    TEDAEnsemble.k=TEDAEnsemble.k+1