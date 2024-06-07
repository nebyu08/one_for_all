import numpy as np

class Relu:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,inputs)
    
    def backward(self,dvalues):
        drelu=dvalues.copy()
        drelu[self.output<=0]=0
        self.drelu=drelu
        return drelu

    

class Softmax:
    def forward(self,inputs):
        #we are scaling the value
        self.scaled=np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        #lets get probability distribution
        self.outputs=self.scaled/np.sum(self.scaled,axis=1,keepdims=True)

    def backward(self,dvalues):
        self.dinputs=np.zeros_like(dvalues)

        #lets loop over the matrix of inputs and operate on the vector values
        for index,(single_outputs,single_dvalues) in enumerate(zip(self.outputs,dvalues)):
            single_outputs=single_outputs.reshape(-1,1)  #change it into column vector
            jackobian_matrix=np.diagflat(single_outputs) - np.dot(single_outputs,single_outputs.T)
           # print(f"here we are:{single_outputs.shape} and {jackobian_matrix.shape}")
            self.dinputs[index]=np.dot(jackobian_matrix,single_dvalues)