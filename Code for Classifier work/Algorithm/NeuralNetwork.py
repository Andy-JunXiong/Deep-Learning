'''
Created on 26 Apr. 2018

@author: shifangzhou & CharlesHe
'''
import numpy as np
import h5py
import random
#Please select beginMode "predict" or "validation". If "predict", the test data file will be loaded and print classification results.
#If "validation", the test data will be selected from the trainning set and print accuracy for each epoch.
beginMode = "predict"

# This module is for forward feeding batch_normalization, with 'trainning' and 'testing' mode
# x is the input mini-batch data set and for every batch_normalization layer gamma and beta
# (representing the scale and shift) are initialized to be one and zero respectively. The
# parameter term is used to modify momentum term and the training or testing mode.
# an output is passed to the activation function outside the batch-normalization layer and important
# information is stored in cache in an effort to be utilized in the back propagation.
def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    D, N = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_mean = np.transpose([running_mean])
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    running_var = np.transpose([running_var])

    if mode == 'training':

        out, cache = None, None
        sample_mean = np.mean(x, axis=1)
        sample_mean = np.transpose([sample_mean])
        sample_var = np.var(x, axis=1)
        sample_var = np.transpose([sample_var])
        out_ = (x - sample_mean) / np.sqrt(sample_var + eps)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        out = gamma * out_ + beta
        cache = (out, x, sample_var, sample_mean, eps, gamma, beta) #cache all useful inforamtion for next iteration

    elif mode == 'testing':
        out, cache = None, None
        scale = gamma / np.sqrt(running_var + eps)
        out = x * scale + (beta - running_mean * scale)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

#This module is for Batching Normalization when doing back propagation
# dout is the derivative from the upper layer and cache is generated in the forward feed process
# the return value dx is the derivative be passed to the downside layer and dgamma are dbeta
# are useful for updating the scale and shift for the next forward batch
def batchnorm_backward(dout, cache):

    out_, x, sample_var, sample_mean, eps, gamma, beta = cache
    N = x.shape[1]
    dout_ = gamma * dout
    dvar = np.sum(dout_ * (x - sample_mean) * -0.5 * (sample_var + eps) ** -1.5, axis=1)
    dvar = np.transpose([dvar])
    dx_ = 1 / np.sqrt(sample_var + eps)
    dvar_ = 2 * (x - sample_mean) / N

    # intermediate for convenient calculation
    di = dout_ * dx_ + dvar * dvar_
    dmean = -1 * np.sum(di, axis=1)
    dmean = np.transpose([dmean])
    dmean_ = np.ones_like(x) / N

    dx = di + dmean * dmean_
    dgamma = np.sum(dout * out_, axis=1)
    dbeta = np.sum(dout, axis=1)

    return dx, dgamma, dbeta

##SoftMax function attached to output layer
#@param  a matrix including a mini-batch's outputs from the output layer
#@return  a same dimension matrix
def SoftMax(outputs):

    arr = np.array(outputs)
    exps = np.exp(arr-np.max(arr))
    return exps / np.sum(exps, axis = 0)


##this is the cross entropy loss derivation which calculate the gradients for outputs of the output layer
#@param   a matrix including a mini-batch's outputs from the output layer
#@param   a mini-batch's labels vector
#return   the gradients for outputs of the output layer
def Loss_derivation(outputs, Y):
    Y = np.array(Y)
    data = Y.shape[0]
    grad = SoftMax(outputs)
    for i in range(0, outputs.shape[1]):
        grad[Y[i], i] = grad[Y[i], i]-1
    grad = grad / data
    return grad

##This is cross entropy function module in order to calculate loss of each iteration
#@param  a matrix including a mini-batch's outputs from the output layer
#@param  a mini-batch's labels vector
#@param  a weight list including each layer's weight matrix
#@param  lambd is a weight decay rate
#@return loss for this mini-batch training
def cross_entropy(outputs, Y, W, lambd):
    Y = np.array(Y)
    data = Y.shape[0]
    probs = SoftMax(outputs)
    loss = []
    for i in range(0, probs.shape[1]):
        loss.append(-np.log(probs[Y[i], i]))
    lossTotal = np.sum(loss) / data
    lossTotal = weight_decay(W, outputs, lambd, lossTotal) #apply weight decay to the loss function
    return lossTotal

##Relu activation for each layer's neurons
#@param a matrix for W.dot(X)
#@return a matrix after Relu's filtering
def Relu(net):
    return np.maximum(0, net)

##Relu derivative function module
#@param  a matrix for W.dot(X)
#@return  a Relu's derivative matrix
def Relu_derivative(net):
    dev_matrix = np.ones((net.shape[0], net.shape[1]))
    for i in range(0, net.shape[0]):
        for j in range(0, net.shape[1]):
            if i < 0:
                dev_matrix[i, j] = 0
            else:
                dev_matrix[i, j] = 1

    return dev_matrix


class NeuralNetwork:
    #Initialization of a neural network
    def __init__(self, layerUnitsNum):#layerUnitsNum is a list which contains the number of neurons in each layer. This includes input layers, hidden layers and output layer
        self.activation = Relu
        self.activation_derivative = Relu_derivative
        self.weights = []
        self.biases = []
        self.layer = layerUnitsNum
        self.velocity = [] #first order momentum
        self.adam=[] #second order momentum for Adam optimiser
        self.gamma = [] #scale for batch normalization
        self.belta = [] #shift for batch normalization
        for i in range(1, len(layerUnitsNum)):
            #weights,biases initialization
            self.weights.append(np.random.randn(layerUnitsNum[i], layerUnitsNum[i-1]) * np.sqrt(2 / (layerUnitsNum[i-1] + 1)))
            self.biases.append(np.random.randn(layerUnitsNum[i], 1))
            self.velocity.append(np.zeros((layerUnitsNum[i], layerUnitsNum[i-1])))
            self.adam.append(np.zeros((layerUnitsNum[i], layerUnitsNum[i-1])))
            self.gamma.append(np.ones(layerUnitsNum[i]))
            self.belta.append(np.zeros(layerUnitsNum[i]))
    ##The main train process including forward propagation and back propagation is implemented here
    #@param train data set without label
    #@param labels for train data
    #@param alpha(learning rate)
    #@param num of epochs
    #@param  mini_batch_size
    #@param v_decay(momentum)
    #@param adamBelta(second order momentum for Adam optimiser
    #@param  dropout_prob (drop out probability)
    #@param  lamb for weight decay
    def train(self, xData, yData, alpha, epochs, mini_batch_size,v_decay,adamBelta,dropout_prob, lamb, eps=1e-5):
        test = 0
        x = np.atleast_2d(xData)
        x = np.array(x)
        y = np.array(yData)
        inputs = []
        outputs = []
        first_time = 0
        for i in range(epochs):
            # shuffle the data
            indices = np.random.permutation(xData.shape[0])
            x = x[indices]
            y = y[indices]
            for k in range(0, xData.shape[0], mini_batch_size): #divide trainning data into mini-batch
                x_mini = x[k:k+mini_batch_size]
                y_mini = y[k:k+mini_batch_size]
                if k+mini_batch_size>xData.shape[0]:
                    x_mini=x[k:xData.shape[0]]
                    y_mini=y[k:xData.shape[0]]

                first_inputs = x_mini.T
                net = []
                inputs = []
                died = []
                caches = []
                inputs.append(first_inputs)
                #forword propagation
                for w in range(0, len(self.weights)):
                    bn_param = {'mode': 'training'}
                    temp = np.dot(self.weights[w], inputs[-1]) + np.array([self.biases[w][:, 0]]).T
                    if first_time == 0:
                        self.gamma[w] = np.array([self.gamma[w]])
                        self.belta[w] = np.array([self.belta[w]])
                    test += 1
                    #do forward batch normalization
                    temp1, cache = batchnorm_forward(temp, self.gamma[w].T, self.belta[w].T, bn_param)
                    caches.append(cache)
                    net.append(temp1)

                    #drop out some neurons and stored died neurons' index for back propagation
                    if w != len(self.weights) - 1:
                        net[-1], diedi = dropout(net[-1], dropout_prob[w])
                        died.append(diedi)
                    outputs = self.activation(net[-1])
                    inputs.append(outputs)

                probs = SoftMax(outputs)
                loss = cross_entropy(outputs, y_mini, self.weights, lamb)
                delta = Loss_derivation(outputs, y_mini)*Relu_derivative(net[len(self.layer)-2])
                #delta is the sensitivity for each layer
                weights_grads = []
                biases_grads = []
                gamma_grads = []
                belta_grads = []
                cachesIndex = 1
                #do back propagation
                for l in range(len(self.layer)-2, -1, -1):
                    #do backward batch normalization
                    delta, dgamma, dbelta = batchnorm_backward(delta, caches[len(caches) - cachesIndex])
                    cachesIndex += 1
                    gamma_grads.append(dgamma)
                    belta_grads.append(dbelta)
                    wdelta = inputs[l].dot(delta.T)
                    #backward weight decay
                    wdelta = weight_decay_back(wdelta, self.weights[l].T, inputs[l], lamb)
                    bdelta = delta

                    #momentum and Adam implements here
                    self.velocity[l] = self.velocity[l]*v_decay+0.001*wdelta.T
                    self.adam[l] = np.maximum(self.adam[l], self.adam[l]*adamBelta+(1-adamBelta)*np.power(wdelta.T,2))

                    weights_grads.append(self.velocity[l])
                    biases_grads.append(bdelta)

                    #implement backward drop out
                    if l != len(self.layer) - 2:
                        for t in range(0, delta.shape[0]):
                            delta[t] = delta[t] * died[l][t]
                        delta /= (1 - dropout_prob[l])

                    if l == 0:
                        delta = delta
                    else:
                        #pass sensitivity back
                        delta = self.weights[l].T.dot(delta)*Relu_derivative(net[l-1])


                weights_grads.reverse()
                biases_grads.reverse()
                gamma_grads.reverse()
                belta_grads.reverse()
                #update weights,biases,gamma and beta
                for u in range(len(self.weights)):
                    self.weights[u] = self.weights[u]-weights_grads[u]/(np.sqrt(self.adam[u])+eps)
                    self.weights[u] = self.weights[u]-weights_grads[u]
                    self.biases[u] = np.array([self.biases[u][:,0]]).T-alpha*biases_grads[u]
                    self.gamma[u] = self.gamma[u]-alpha*gamma_grads[u]
                    self.belta[u] = self.belta[u]-alpha*belta_grads[u]
                first_time += 1
            #print accuracy for each epoch
            if beginMode == "validation":
                print("epoch: "+str(i))
                result = self.predict(xData)
                self.testAccuracy(result,yData)

    ##This module is to calculate accuracy
    #@param results(predict results)
    #@param datay(data labels)
    def testAccuracy(self,results, datay):
        #datay=np.array(datay)
        results.tolist()
        datay.tolist()
        correct = 0
        for i in range(0, len(results)):
            if results[i] == datay[i]:
                correct += 1
        accuracy = float(correct/len(results))
        print(accuracy)

    ##This module is to predict a test data after training
    #@param datax(test data)
    #@return result(predict results)
    def predict(self, datax):
        datax = np.array(datax)
        datax = np.atleast_2d(datax)
        datax = datax.T
        for l in range(0, len(self.weights)):
            datax = np.dot(self.weights[l], datax) + np.transpose([self.biases[l][:, 0]])
            bn_param = {'mode': 'testing'}
            datax, cache = batchnorm_forward(datax, self.gamma[l].T, self.belta[l].T, bn_param)
            datax = Relu(datax)
        result = np.argmax(datax,axis=0)
        return result

##This module is drop out some randomly selected neurons for each layer to avoid overfitting
#@param x (a matrix for each layer calculation of W.dot(Y))
#@param prob(drop out probability)
#@return x (a matrix filtered by dropout)
#@return sample (a random generated dropout vector such as [1,0,0,1,0,...,1], 0 for died neuron, 1 for retained neuron)
def dropout(x, prob):
    retain_prob = 1 - prob
    sample = np.random.binomial(n=1, p=retain_prob, size=x.shape[0])
    for i in range(0, sample.shape[0]):
        x[i] = x[i] * sample[i]
    x /= retain_prob
    return x, sample


# x the data set and W is the weight matrix for this layer and lambda is the parameter which
# indicates the degree for decaying and cost is the cost without weight decay
# The return value is the new cost
def weight_decay(W, x, lambd, cost):
    m = x.shape[1]
    sum = 0
    for i in range(0,len(W)):
        sum = sum + np.sum(np.square(W[i]))
    L2_regularization_cost = sum * (lambd / (2 * m))
    return cost + L2_regularization_cost

# dW is the derivative for updating W matrix and W is the weights before updating
# x is the dataset and lambda shows the degree for decaying
# The return value is the new updated weight with weight_decay
def weight_decay_back(dW, W, x, lambd):
    m = x.shape[1]
    return dW + (lambd/m) * W


if __name__ == "__main__":

    with h5py.File('train_128.h5', 'r') as H:
        data = np.copy(H['data'])
    with h5py.File('train_label.h5', 'r') as H:
        label = np.copy(H['label'])
    # with h5py.File('Predicted_labels.h5','r') as H:
    #     predicts = np.copy(H['label'])
    #     print(predicts)

    network = NeuralNetwork([128, 70, 70, 10])
    if beginMode == "predict":
        with h5py.File('test_128.h5','r') as H:
            test = np.copy(H['data'])
            network.train(data, label, 0.0001, 35, 64, 0.9, 0.999, [0.2, 0.5], 0.00001, 1e-5)
            result = network.predict(test)
            result = np.array(result)
            # f = h5py.File('Predicted_labels.h5','w')#load predict data to h5 file
            # f.create_dataset('label',data=result)
            # f.close()
            print(result)
    else:
        test = data[0:6000]
        testLabel = label[0:6000]
        data = data[6001:60000]
        label = label[6001:60000]
        #xData, yData, alpha(learning rate), epochs, mini_batch_size, v_decay(momentum), adamBelta, dropout_prob, lamb(weight decay), eps=1e-5
        network.train(data, label, 0.0001, 35, 64, 0.9, 0.999, [0.2, 0.5], 0.00001, 1e-5)
