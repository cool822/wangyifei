import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_daoshu(x):
    return sigmoid(x)*(1-sigmoid(x))
def softmax(x):
    exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
    return exp_x/np.sum(exp_x,axis=1,keepdims=True)
def relu(x):
    return np.max(x,0)
def relu_daoshu(x):
    if x>0:
        return 1
    else:
        return 0
def CE(y_true,y_pred):
    return -np.sum(y_true*np.log(y_pred))
class NN:
    def __init__(self,input_size,hidden_size,output_size,activation,lambda2):
        self.activation=activation
        self.w1=np.random.normal(input_size,hidden_size)
        self.b1=np.zeros((1,hidden_size))
        self.w2=np.random.normal(hidden_size,output_size)
        self.b2=np.zeros((1,output_size))
        self.lambda2=lambda2
        self.loss=float('inf')
        self.w1_best=self.w2_best=self.b1_best=self.b2.best=0
    def forward(self,x):
        self.Z1=np.dot(x,self.w1)+self.b1
        self.A1=self.activation(self.Z1)
        self.Z2=np.dot(self.A1,self.w2)+self.b2
        self.A2=softmax(self.Z2)
        return self.A2
    def forward1(self,x):
        self.Z1=np.dot(x,self.w1_best)+self.b1_best
        self.A1=self.activation(self.Z1)
        self.Z2=np.dot(self.A1,self.w2_best)+self.b2_best
        self.A2=softmax(self.Z2)
        return self.A2
    def compute_loss(self,y_true,y_pred):
        m=y_pred.shape[0]
        cross_entropy_loss=CE(y_true,y_pred)
        l2_penalty=(self.lambda2/(2*m))*(np.sum(np.square(self.w1))+np.sum(np.square(self.w2)))
        return cross_entropy_loss+l2_penalty
    def backward(self,x,y,learning_rate):
        m=y.shape[0]
        output_gap=self.A2-y#交叉熵损失和softmax求导
        dw2=np.dot(self.A2.T,output_gap)/m
        db2=np.sum(output_gap,axis=0,keepdims=True)
        hidden_gap=np.dot(output_gap,self.w2.T)*sigmoid_daoshu(self.Z1)
        dw1=np.dot(x.T,hidden_gap)/m
        db1=np.sum(hidden_gap,axis=0,keepdims=True)
        #SGD更新
        self.w2-=learning_rate*dw2
        self.b2-=learning_rate*db2
        self.w1-=learning_rate*dw1
        self.b1-=learning_rate*db1
    def train(self,X,y,epochs,learning_rate):
        for epoch in range(epochs):
            y_pred=self.forward(X)
            loss=self.compute_loss(y,y_pred)
            if loss<self.loss:
                self.loss=loss
                self.w1_best=self.w1
                self.w2_best=self.w2
                self.b1_best=self.b1
                self.b2_best=self.b2
            self.backward(X,y,learning_rate)
            if epoch%10==0:
                print(f'Epoch:{epoch},Loss:{loss}')
    def get_parameters(self):
        return self.w1_best,self.b1_best,self.w2_best,self.b2_best
    def test(self,X,y):
        y_pred=self.forward(X)
        accuracy=np.mean(np.argmax(y_pred,axis=1)==np.argmax(y,axis=1))
        print(f'Accuracy:{accuracy}')







