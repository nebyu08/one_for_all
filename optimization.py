import numpy as np

class SGD:
    def __init__(self,lr=0.01,decay_rate=0) -> None:
        self.learning_rate=lr
        self.iterations=0
        self.decay_rate=decay_rate
    

    def preupdate_params(self):
        if self.decay_rate:
            self.learning_rate=self.learning_rate/(1+self.decay_rate*self.iterations)
    
    
    def update_params(self,layer):
        layer.weights+=self.learning_rate*layer.dweights
        layer.bias+=self.learning_rate*layer.dbias
    
    def postupdate_params(self):
        self.iterations+=1
        
