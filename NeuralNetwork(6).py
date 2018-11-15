'''
Created on 26 Apr. 2018

@author: shifangzhou
'''
import numpy as np
import h5py


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    D, N = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_mean = np.transpose([running_mean])
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    running_var = np.transpose([running_var])

    out, cache = None, None

    if mode == 'train':

        sample_mean = np.mean(x, axis=1)
        sample_mean = np.transpose([sample_mean])
        sample_var = np.var(x, axis=1)
        sample_var = np.transpose([sample_var])
        out_ = (x - sample_mean) / np.sqrt(sample_var + eps)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        out = gamma * out_ + beta
        cache = (out, x, sample_var, sample_mean, eps, gamma, beta)

    elif mode == 'test':
        scale = gamma / np.sqrt(running_var + eps)
        out = x * scale + (beta - running_mean * scale)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):

    dx, dgamma, dbeta = None, None, None

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


def SoftMax(outputs):

    arr = np.array(outputs)
    exps = np.exp(arr-np.max(arr))
    return exps / np.sum(exps, axis = 0)


def Loss_derivation(outputs, Y):#this is the cross entropy loss derivation
    Y = np.array(Y)
    data = Y.shape[0]
    grad = SoftMax(outputs)
    for i in range(0, outputs.shape[1]):
        grad[Y[i], i] = grad[Y[i], i]-1
    grad = grad / data
    return grad


def cross_entropy(outputs, Y, W, lambd):
    Y = np.array(Y)
    data = Y.shape[0]
    probs = SoftMax(outputs)
    loss = []
    for i in range(0, probs.shape[1]):
        loss.append(-np.log(probs[Y[i], i]))
    lossTotal = np.sum(loss) / data
    lossTotal = weight_decay(W, outputs, lambd, lossTotal)
    return lossTotal


def Relu(net):
    return np.maximum(0, net)


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
    def __init__(self, layerUnitsNum):#layerUnitsNum is a list which contains the number of neurons in each layer. This includes input layers, hidden layers and output layer
        self.activation = Relu
        self.activation_derivative = Relu_derivative
        self.weights = []
        self.biases = []
        self.layer = layerUnitsNum
        self.velocity = []
        self.adam=[]
        self.gamma = []
        self.belta = []
        for i in range(1, len(layerUnitsNum)):#weights,biases initialization
            self.weights.append(np.random.randn(layerUnitsNum[i], layerUnitsNum[i-1]) * np.sqrt(2 / (layerUnitsNum[i-1] + 1)))
            self.biases.append(np.random.randn(layerUnitsNum[i], 1))
            self.velocity.append(np.zeros((layerUnitsNum[i], layerUnitsNum[i-1])))
            self.adam.append(np.zeros((layerUnitsNum[i], layerUnitsNum[i-1])))
            self.gamma.append(np.ones(layerUnitsNum[i]))
            self.belta.append(np.zeros(layerUnitsNum[i]))

    def train(self, xData, yData, alpha, epochs, mini_batch_size, v_decay,adamBelta,dropout_prob, lamb, eps=1e-5):
        test = 0
        x = np.atleast_2d(xData)
        x = np.array(x)
        y = np.array(yData)
        inputs = []
        outputs = []
        first_time = 0
        for i in range(epochs):
            for k in range(0, xData.shape[0], mini_batch_size):
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
                for w in range(0, len(self.weights)):#forword propagation
                    bn_param = {'mode': 'train'}
                    temp = np.dot(self.weights[w], inputs[-1]) + np.array([self.biases[w][:,0]]).T
                    if first_time == 0:
                        self.gamma[w] = np.array([self.gamma[w]])
                        self.belta[w] = np.array([self.belta[w]])

                    #print(test, end=" ", flush=True)
                    test += 1
                    temp1, cache = batchnorm_forward(temp, self.gamma[w].T, self.belta[w].T, bn_param)
                    caches.append(cache)
                    #if w != len(self.weights)-1:
                    net.append(temp1)
                    net[-1],diedi = dropout(net[-1],dropout_prob)
                    died.append(diedi)
                    outputs = self.activation(net[-1])


                    inputs.append(outputs)
                probs = SoftMax(outputs)
                loss = cross_entropy(outputs, y_mini, self.weights, lamb)
                #print("loss")
                #print(loss)
                delta = Loss_derivation(outputs, y_mini)*Relu_derivative(net[len(self.layer)-2])
                #delta is the similarity for each layer
                weights_grads = []
                biases_grads = []
                gamma_grads = []
                belta_grads = []
                cachesIndex = 1

                for l in range(len(self.layer)-2,-1,-1):#back propagation
                    delta, dgamma, dbelta=batchnorm_backward(delta, caches[len(caches) - cachesIndex])
                    cachesIndex += 1
                    gamma_grads.append(dgamma)
                    belta_grads.append(dbelta)
                    wdelta = inputs[l].dot(delta.T)
                    wdelta = weight_decay_back(wdelta, self.weights[l].T, inputs[l], lamb)
                    bdelta = delta

                    #momentum gradient implements here

                    self.velocity[l] = self.velocity[l]*v_decay+0.001*wdelta.T
                    self.adam[l] = np.maximum(self.adam[l],self.adam[l]*adamBelta+(1-adamBelta)*np.power(wdelta.T,2))

                    weights_grads.append(self.velocity[l])
                    biases_grads.append(bdelta)

                    if l != len(self.layer)-2:
                        for t in range(0,delta.shape[0]):
                            delta[t] = delta[t] * died[l][t]
                        delta /= (1-dropout_prob)

                    if l == 0:
                        delta = delta
                    else:
                        delta = self.weights[l].T.dot(delta)*Relu_derivative(net[l-1])


                weights_grads.reverse()
                biases_grads.reverse()
                gamma_grads.reverse()
                belta_grads.reverse()
                for u in range(len(self.weights)):

                    self.weights[u] = self.weights[u]-weights_grads[u]/(np.sqrt(self.adam[u]+eps))
                    self.weights[u] = self.weights[u]-weights_grads[u]
                    self.biases[u] = np.array([self.biases[u][:,0]]).T-alpha*biases_grads[u]
                    self.gamma[u] = self.gamma[u]-alpha*gamma_grads[u]
                    self.belta[u] = self.belta[u]-alpha*belta_grads[u]
                first_time += 1
            print("epoch: "+str(i))
            self.predict(xData,yData)

    def predict(self, datax, datay):
        datax = np.array(datax)
        datax = np.atleast_2d(datax)
        datax = datax.T
        datay=np.array(datay)
        for l in range(0, len(self.weights)):
            datax = np.dot(self.weights[l], datax) + np.transpose([self.biases[l][:, 0]])
            bn_param = {'mode': 'test'}
            datax, cache = batchnorm_forward(datax, self.gamma[l].T, self.belta[l].T, bn_param)
            datax = Relu(datax)

        result = np.argmax(datax,axis=0)
        correct = 0
        for i in range(0, result.shape[0]):
            if result[i] == datay[i]:
                # print("result: "+ str(result[i])+" actual: "+str(datay[i]))
                correct += 1
        accuracy = float(correct/datax.shape[1])
        print(accuracy)
        #return data


def dropout(x,prob):
    retain_prob = 1-prob
    sample = np.random.binomial(n=1, p=retain_prob, size=x.shape[0])
    for i in range(0, sample.shape[0]):
        x[i] = x[i]*sample[i]
    x /= retain_prob
    return x, sample


def weight_decay(W, x, lambd, cost):
    m = x.shape[1]
    sum = 0
    for i in range(0,len(W)):
        sum = sum + np.sum(np.square(W[i]))
    L2_regularization_cost = sum * (lambd / (2 * m))
    return cost + L2_regularization_cost


def weight_decay_back(dW, W, x, lambd):
    m = x.shape[1]
    return dW + (lambd/m) * W


if __name__ == "__main__":

    with h5py.File('train_128.h5','r') as H:
        data = np.copy(H['data'])
    with h5py.File('train_label.h5','r') as H:
        label = np.copy(H['label'])
    #with h5py.File('')
    test = data[10000:20000]
    #print(test.shape)
    testLabel = label[10000:20000]
    network = NeuralNetwork([128, 75, 75, 10])
    data = data[20000:60000]
    label = label[20000:60000]
    #xData, yData, alpha(learning rate), epochs, mini_batch_size, v_decay(momentum), adamBelta, dropout_prob, lamb(weight decay), eps=1e-5
    network.train(data, label, 0.0001, 10, 32, 0.9, 0.9, 0.5, 0.00001, 1e-5)
    network.predict(test, testLabel)
