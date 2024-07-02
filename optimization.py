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
            self.current_learning_rate=self.learning_rate/(1+self.decay_rate*self.iterations)
    
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
