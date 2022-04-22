import numpy
import math
import benchmarks as Fit    
import matplotlib.pyplot as plt  

def alpha_new(alpha,NGen):
    #% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
    #% alpha_0=0.9
    delta=1-(10**(-4)/0.9)**(1/NGen);
    alpha=(1-delta)*alpha
    return alpha


# General parameters
lb=-10;    
ub=10;
MaxGeneration=1000;
dim=30;
n=50;

#FFA parameters
alpha=0.5  # Randomness 0--1 (highly random)
betamin=0.20  # minimum value of beta
gamma=1   # Absorption coefficient
if not isinstance(lb, list):
   lb = [lb] * dim
if not isinstance(ub, list):
   ub = [ub] * dim
   
zn=numpy.ones(n)
zn.fill(float("inf")) 
    
ns = numpy.zeros((n, dim))
for i in range(dim):
   ns[:, i] = numpy.random.uniform(0,1, n) * (ub[i] - lb[i]) + lb[i]
Lightn=numpy.ones(n)
Lightn.fill(float("inf")) 
    
convergence=[]
 
# Main loop
for k in range (0,MaxGeneration):     # start iterations
    
 #% This line of reducing alpha is optional
 alpha=alpha_new(alpha,MaxGeneration);
        
 #% Evaluate new solutions (for all n fireflies)
 for i in range(0,n):
   zn[i]=Fit.F1(ns[i,:])
   Lightn[i]=zn[i]
        
        
                
        
 # Ranking fireflies by their light intensity/objectives
    
        
 Lightn=numpy.sort(zn)
 Index=numpy.argsort(zn)
 ns=ns[Index,:]
        
        
 #Find the current best
 nso=ns
 Lighto=Lightn
 nbest=ns[0,:] 
 Lightbest=Lightn[0]
 #% For output only
 fbest=Lightbest;
        
 #% Move all fireflies to the better locations
 scale = []
 for b in range(dim):
   scale.append(abs(ub[b] - lb[b]))
 scale = numpy.array(scale)
 for i in range (0,n):
   # The attractiveness parameter beta=exp(-gamma*r)
   for j in range(0,n):
      r=numpy.sqrt(numpy.sum((ns[i,:]-ns[j,:])**2));
      # Update moves
      if Lightn[i]>Lighto[j]: # Brighter and more attractive
         beta0=1
         beta=(beta0-betamin)*math.exp(-gamma*r**2)+betamin
         tmpf=alpha*(numpy.random.rand(dim)-0.5)*scale
         ns[i,:]=ns[i,:]*(1-beta)+nso[j,:]*beta+tmpf
        
        
        
 convergence.append(fbest)
        	
 IterationNumber=k
 BestQuality=fbest
        
 print(['At iteration '+ str(k)+ ' the best fitness is '+ str(BestQuality)])

# Data for plotting
plt.plot(convergence)
plt.xlabel('Iteration') 
plt.ylabel('Cost') 
plt.title('Cost Function in Itterations') 
  
# function to show the plot 
plt.show()  
    
    
    
    
    
