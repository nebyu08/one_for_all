import numpy as np

class SGD:
    def __init__(self,lr=0.01,decay_rate=0,momentum=0) -> None:
        self.learning_rate=lr
        self.iterations=0
        self.decay_rate=decay_rate
        self.current_learning_rate=lr
        self.momentum=momentum
    
    #this is for the learning rate
    def pre_update_params(self):
        if self.decay_rate:
            self.current_learning_rate=self.learning_rate * 1/(1+self.decay_rate*self.iterations)
    
    #this is for the weght update with momentum
    def update_params(self,layer):

        #we are initializing the momentum for the gradient of the weight and bias
        if not hasattr(layer,'weight_momentums'):
            self.weight_momentums=np.zeros_like(layer.weights)
            self.bias_momentums=np.zeros_like(layer.bias)

            #we are calculating the amount to which we should change our current gradient
            self.weight_update=self.momentum*self.weight_momentums - self.learning_rate*layer.dweights
            self.bias_update=self.momentum*self.bias_momentums - self.learning_rate*layer.dbias

            #lets assign the current calculated weight and bias as the next caculation intial accumulated gradient
            self.weight_momentums=self.weight_update
            self.bias_momentums=self.bias_update
        
        else:
            self.weight_update=-self.current_learning_rate*layer.dweights
            self.bias_update=-self.current_learning_rate*layer.dbias

        #updating the weights and bias after backpropagation 
        layer.weights+=self.weight_update
        layer.bias+=self.bias_update

    #this is for the learning rate
    def post_update_params(self):
        self.iterations+=1

class Ada_Grad:
    """
    this is the implementation of Ada grad optimizer from scratch.
    """
    def __init__(self,lr=0.01,decay_rate=0.1,epsilon=1e-7):
        self.decay_rate=decay_rate
        self.current_learning_rate=lr
        self.learning_rate=lr
        self.epsilon=epsilon
        self.iteration=0

    def pre_update_params(self):
        if self.decay_rate:
            self.current_learning_rate=self.learning_rate*1/(1+self.decay_rate*self.iteration)
    
    def update_params(self,layer):
        if not hasattr(layer,"weight_cache"):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.bias)
        
        #must store the previous value of gradient for comparision sake
        layer.weight_cache+=(layer.dweights**2)
        layer.bias_cache+=(layer.dbias**2)


        layer.weights+=-self.learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.bias+=-self.learning_rate*layer.dbias/(np.sqrt(layer.bias_cache)+self.epsilon)
    
    def post_update_params(self):
        self.iteration+=1

class RMS_Prop:
    def __init__(self,lr=0.001,rho=0.9,decay_rate=0,epsilon=1e-7):
        self.learnig_rate=lr
        self.current_learning_rate=lr
        self.rho=rho
        self.epsilon=epsilon
        self.decay_rate=decay_rate
        self.iteration=0
    
    def pre_update_params(self):
        if self.decay_rate:
            self.current_learning_rate=self.learnig_rate*(1/(1+self.decay_rate*self.iteration))

    def update_params(self,layer):
        if not hasattr(layer,"weight_cache"):
            layer.cache_weight=np.zeros_like(layer.weights)
            layer.cache_bias=np.zeros_like(layer.bias)
        
        #calculating the cahe of the gradient
        layer.cache_weight=self.rho*layer.cache_weight+ (1-self.rho) *layer.dweights**2
        layer.cache_bias=self.rho*layer.cache_bias+ (1-self.rho) *layer.dbias**2

        #lets update the parameters of the model
        layer.weights+=-self.current_learning_rate*layer.dweights/(np.sqrt(layer.cache_weight)+self.epsilon)
        layer.bias+=-self.current_learning_rate*layer.dbias/(np.sqrt(layer.cache_bias) + self.epsilon)

    def post_update_params(self):
        self.iteration+=1


        