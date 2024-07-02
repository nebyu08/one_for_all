import numpy as np

class Loss:
    def calculate(self,y_pred,y_true):
        self.loss=self.forward(y_pred,y_true)
        self.avg_loss=np.mean(self.loss)
        return self.avg_loss

class Categorical_loss_entropy(Loss):
    def __init__(self):
        super().__init__()
    def forward(self,y_pred,y_true):
        y_pred_cliped=np.clip(y_pred,1e-7,1-1e-7)
        #check for the type of actual values
        if len(y_true.shape)==2:
            correctness=np.sum(y_pred_cliped*y_true,axis=1)
            
        elif len(y_true.shape)==1:
            correctness=y_pred_cliped[range(len(y_pred_cliped)),y_true]
        else:
            raise Except("there is an error with the shape of the input.")
            
        self.loss=-np.log(correctness)
        return self.loss