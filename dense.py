import numpy as np

class Dense:
    """
    this is the structure of a single hidden layer that is meant to be customized
    """
    def __init__(self,n_inputs,n_neurons):
        #self.inputs=n_inputs
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))
    
    def forward(self,inputs):
        """ this is the feedforward opertaion of a neural network."""
        self.inputs=inputs
        self.outputs=np.dot(inputs,self.weights)+self.bias
        return self.outputs
    
    def backward(self,dvalues):
        #print(f"hi:self.inputs:{self.inputs}")
        self.dinputs=np.dot(dvalues,self.weights.T)
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dbias=np.sum(dvalues,axis=0,keepdims=True)
