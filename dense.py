import numpy as np

class Dense:
    """
    this is the structure of a single hidden layer that is meant to be customized
    """
    def __init__(self,n_inputs,n_neurons,regularizer1_weight=0,regularizer1_bias=0,regularizer2_weight=0,regularizer2_bias=0):
        #self.inputs=n_inputs
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))
        self.regularizer1_weight=regularizer1_weight
        self.regularizer1_bias=regularizer1_bias
        self.regularizer2_weight=regularizer2_weight
        self.regularizer2_bias=regularizer2_bias
    
    def forward(self,inputs):
        """ this is the feedforward opertaion of a neural network."""
        self.inputs=inputs
        self.outputs=np.dot(inputs,self.weights)+self.bias
        return self.outputs
    
    def regularization_loss(self,layer):
        self.regularizer_loss=0

        #setting up the regularizer for the l1 type
        if self.regularizer1_weight>0:
            #setting up for the weights
            self.regularization_loss+=self.regularizer1_weight*(np.sum(np.abs(layer.weights)))

        #check for the bias of the l1 regularization
        if self.regularizer1_bias>0:
            #lets setup for the bias
            self.regularization_loss+=self.regularizer1_bias*(np.sum(np.abs(layer.bias)))
        
        #check for the second type or the l2

        #check for the weights
        if self.regularizer2_weight:
            #adding the weights of the layer
            self.regularization_loss+=self.regularizer2_weight*(np.sum(layer.weights*layer.weights))

        #check for the bias
        if self.regularizer2_bias:
            #adding the bias of the layer
            self.regularization_loss+=self.regularizer2_bias*(np.sum(layer.bias*layer.bias))
        
        return self.regularization_loss
    
    def backward(self,dvalues):
        #print(f"hi:self.inputs:{self.inputs}")
        self.dinputs=np.dot(dvalues,self.weights.T)
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dbias=np.sum(dvalues,axis=0,keepdims=True)

        #check for the weight1
        
        #this is for the regularization of type 1 (L1)
        if self.regularizer1_weight>0:
            dl1=np.ones_like(self.weights)
            dl1[self.weights<0]=-1
            self.dweights+=self.regularizer1_weight*dl1
        #type 2 regularization
        if self.regularizer2_weight>0:
            self.dweights+=2*self.regularizer2_weight*self.weights

        #for the l2 regularization

        #lets update the biases
        if self.regularizer1_bias>0:
            dl1=np.ones_like(self.bias)
            dl1[self.bias<0]=-1
            self.dbias+=self.regularizer1_bias*dl1

        if self.regularizer2_bias>0:
            self.dbias+=2*self.regularizer2_bias*self.bias