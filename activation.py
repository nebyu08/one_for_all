import numpy as np

class Relu:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,inputs)
    
    def backward(self,dvalues):
        drelu=np.dvalues.copy()
        drelu[self.output<=0]=0
        return drelu

    

class Softmax:
    def forward(self,inputs):
        #we are scaling the value
        self.scaled=np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        #lets get probability distribution
        self.outputs=self.scaled/np.sum(self.scaled,axis=1,keepdims=True)
    