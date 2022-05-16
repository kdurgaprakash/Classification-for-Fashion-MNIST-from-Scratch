import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.decomposition import PCA

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 75
NUM_OUTPUT = 10

# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases,NUM_HIDDEN_LAYERS,NUM_HIDDEN):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

# Function to calculate accuracy of the Model
def accuracypercent(X, Y, Yhat):
    maxidxyhat = np.argmax(Yhat,axis = 0)
    maxidxy = np.argmax(Y, axis = 0)
    acc = np.mean(maxidxyhat == maxidxy)
    return acc*100

def data_split(X,Y):
    numexam = Y.shape[1]
    batch_split = np.random.permutation(numexam)
    split_pt = int(0.8 * numexam)
    train_split = batch_split[0:split_pt]
    val_split = batch_split[split_pt:]
    train_X = X[:, train_split]
    train_Y = Y[:, train_split]
    val_X = X[:, val_split]
    val_Y = Y[:, val_split] 
    return train_X, train_Y, val_X, val_Y

def forward_prop (x, y, weightsAndBiases,NUM_HIDDEN_LAYERS, NUM_HIDDEN):
    Ws, bs = unpack(weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)
    
    # W1 shape 10,784 
    # W2,3,4 shapes - 10,10
 
    zs = []
    hs = []

    #x shape 784,48000
    # First Layer - append weights and biases to lists
    bs[0] = bs[0].reshape(len(bs[0]),1)
    Z = np.dot(Ws[0],x) + bs[0]
    h = relu(Z)
    zs.append(Z)  
    hs.append(h)

    #Hidden layer weights and biases
    for i in range(1,NUM_HIDDEN_LAYERS):
        bs[i] = bs[i].reshape(len(bs[i]),1)
        Z = np.dot(Ws[i],hs[i-1]) + bs[i]
        h = relu(Z)
        zs.append(Z)
        hs.append(h)

    # Last layer weights and biases
    bs[NUM_HIDDEN_LAYERS] = bs[NUM_HIDDEN_LAYERS].reshape(len(bs[NUM_HIDDEN_LAYERS]),1)
    Z = np.dot(Ws[NUM_HIDDEN_LAYERS],hs[NUM_HIDDEN_LAYERS-1]) + bs[NUM_HIDDEN_LAYERS] 
    h = softmax(Z)
    zs.append(Z)
    hs.append(h)

    #yhat shape 10,48000 
    yhat = hs[NUM_HIDDEN_LAYERS]
    loss = crossentropyloss(x,y,yhat)
    accuracy = accuracypercent(x,y,yhat)
    return loss, accuracy, zs, hs, yhat

# Softmax function for calculating yhat
def softmax(Z):
    e = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    s = np.sum(e, axis = 0, keepdims= True)
    return e / s 

def relu(Z):
    relu = np.maximum(Z,0)
    return relu

def relu_grad(Z):
    relugrad = Z.copy()
    relugrad[relugrad > 0 ] = 1
    relugrad[relugrad <= 0] = 0
    return relugrad

def back_prop (x, y, weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN):
    loss, accuracy , zs, hs, yhat = forward_prop(x, y, weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)
    #x shape - 784,48000 ; y shape - 48000,10; yhat shape - 10,48000

    Ws, bs = unpack(weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)

    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases

    numexam = y.shape[1]
 
    g = (yhat - y) / numexam #10*5
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        if i == 0:
            dJdWs.append(np.dot(g,x.T))   
        else:
            dJdWs.append(np.dot(g,hs[i-1].T))
            
        dJdbs.append(np.sum(g,axis=1))
        if i == 0:
            g=g
        else:
            g = np.dot(Ws[i].T,g)
            g = g * relu_grad(zs[i-1])   
    dJdWs.reverse()
    dJdbs.reverse()  

    # Concatenate gradients
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]) 

# Function to calculate cross entropy loss
def crossentropyloss(X,Y,Yhat):
    #x shape - 784,48000 ; y shape - 10,48000 ; yhat shape - 10,48000
    loss = -1/Y.shape[1]*np.sum(Y * np.log(Yhat))
    return loss

def train (trainX, trainY, weightsAndBiases, testX, testY):
    
    trainX, trainY, valX, valY = data_split(trainX, trainY)

    #48000 for training, 12000 for testing
    split = trainX.shape[1]


    batch_size = 512 #128
    num_epochs = 150 #50
    alpha = 0.4 #0
    learn_rate = 0.05 #0.001
    
    #Take initial randomized weights
    weightsAndBiases = initWeightsAndBiases(NUM_HIDDEN_LAYERS, NUM_HIDDEN)

    trajectory = np.copy(weightsAndBiases)
    
    # Number of gradient update for each epoch
    num_grad_upd = int(split/batch_size)

    # for e in range(num_epochs):
    for epoch in range(num_epochs):
        # TODO: implement SGD.
        # TODO: save the current set of weights and biases into trajectory; this is
        # useful for visualizing the SGD trajectory.
        for r in range(num_grad_upd):

            # Dividing training examples into batches
            #trainxmini - 784*batch_size; trainymini - 10*batch_size
            trainxmini = trainX[:,(batch_size*r):(batch_size*(r+1))]                    
            trainymini = trainY[:,(batch_size*r):(batch_size*(r+1))]

            grad = back_prop(trainxmini, trainymini, weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)

            # weightsAndBiases = weightsAndBiases - learn_rate*grad

            w,b = unpack(weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)
            dw, db = unpack(grad, NUM_HIDDEN_LAYERS, NUM_HIDDEN)

            for i in range(len(w)):
                reg = alpha*w[i]
                w[i] = w[i] - (learn_rate*dw[i])- (((alpha*learn_rate)/trainymini.shape[1]) * w[i])
                b[i] = b[i] - learn_rate*db[i]

            weightsAndBiases = np.hstack([ W.flatten() for W in w ] + [ B.flatten() for B in b ])                             
            
        # train_cost calculation
        batch_cost,batch_acc, batch_zs, batch_hs, batch_yhat = forward_prop(trainxmini, trainymini, weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)
        
        print('loss for epoch',epoch + 1,':',batch_cost)
        trajectory = np.vstack([trajectory,np.copy(weightsAndBiases)])

    test_cost,test_acc, test_zs, test_hs, test_yhat = forward_prop(testX, testY, weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)
    print('test_cost', test_cost )
    print('test_Accuracy',test_acc)

    #Print Optimal hyperparameters and Test CE and accuracy
    print('Optimal alpha:', alpha)
    print('Optimal batch size:', batch_size)
    print('Optimal number of epochs:', num_epochs)
    print('Optimal learning rate:', learn_rate)
    print('Optimal num hidden layers:', NUM_HIDDEN_LAYERS)
    print('Optimal num hidden:', NUM_HIDDEN)

    return weightsAndBiases, trajectory

# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases (NUM_HIDDEN_LAYERS, NUM_HIDDEN):
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def plotSGDPath (trainX, trainY, trajectory):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.

    pca = PCA(n_components = 2)
    zs = pca.fit_transform(trajectory)

    def toyFunction (x1, x2):
        z = [x1,x2]
        wab = pca.inverse_transform(z)
        loss,_, _, _,_ = forward_prop(trainX, trainY, wab,NUM_HIDDEN_LAYERS, NUM_HIDDEN)
        return loss

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    axis1 = np.linspace( min(zs[:,0]), max(zs[:,0]), 150)  # Just an example
    axis2 = np.linspace( min(zs[:,1]), max(zs[:,1]), 150)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))

    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    Xaxis_ = zs[:, 0]  # Just an example
    Yaxis_ = zs[:, 1]  # Just an example
    Zaxis_ = np.zeros(len(Xaxis_))
    for i in range(len(Xaxis)):
        Zaxis_[i] = toyFunction(Xaxis_[i], Yaxis_[i])
    ax.scatter(Xaxis_, Yaxis_, Zaxis_, color='r')
    plt.show()

# def onehotencoder(labels,num_classes):
def onehotencoder(Y,num_classes):
    Yonehot = np.zeros((len(Y), num_classes))
    for i in range(len(Y)):
        for j in range(num_classes):
               if j == Y[i] :
                Yonehot[i][j] = 1
    return Yonehot

def findBestHyperparameters(xtrain,ytrain,xtest,ytest):

    trainX, trainY, valX, valY = data_split(xtrain, ytrain)

    #48000 for training, 12000 for testing
    split = trainX.shape[1]

    #Initializing parameters
    optim_acc = 0
    optim_epoch = 0
    optim_batch = 0
    optim_lr = 0
    optim_alpha = 0
    optim_weightsAndBiases = []

    #Hyperparameter list
    num_hidden_layers_list = [3,4,5,6]
    num_hidden_list = [30,40,50,75]
    batch_size_list = [64,128,256,512]
    num_epochs_list  = [100,150,200]
    alpha_list = [0.1,0.01,0.01,0.4]
    learn_rate_list = [0.001,0.005,0.05]

    # trajectory = []

    #Hyperparameter tuning
    for NUM_HIDDEN_LAYERS in num_hidden_layers_list:
        for NUM_HIDDEN in num_hidden_list:
            for batch_size in batch_size_list:
                for num_epochs in num_epochs_list:
                    for alpha in alpha_list:
                        for learn_rate in learn_rate_list:
                            
                            #Take initial randomized weights
                            weightsAndBiases = initWeightsAndBiases(NUM_HIDDEN_LAYERS, NUM_HIDDEN)
                            
                            trajectory = np.copy(weightsAndBiases)

                            # Number of gradient update for each epoch
                            num_grad_upd = int(split/batch_size)

                            # for e in range(num_epochs):
                            for e in range(num_epochs):
                                # for r in range(num_grad_upd):
                                for r in range(num_grad_upd):

                                    # Dividing training examples into batches
                                    #trainxmini - 784*batch_size; trainymini - 10*batch_size
                                    trainxmini = trainX[:,(batch_size*r):(batch_size*(r+1))]                    
                                    trainymini = trainY[:,(batch_size*r):(batch_size*(r+1))]

                                    grad = back_prop(trainxmini, trainymini, weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)

                                    # weightsAndBiases = weightsAndBiases - learn_rate*grad

                                    w,b = unpack(weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)
                                    dw, db = unpack(grad, NUM_HIDDEN_LAYERS, NUM_HIDDEN)

                                    for i in range(len(w)):
                                        reg = alpha*w[i]
                                        w[i] = w[i] - (learn_rate*dw[i])- (((alpha*learn_rate)/trainymini.shape[1]) * w[i])
                                        b[i] = b[i] - learn_rate*db[i]

                                    weightsAndBiases = np.hstack([ W.flatten() for W in w ] + [ B.flatten() for B in b ])                             
                                    
                                    batch_cost,batch_acc, batch_zs, batch_hs, batch_yhat = forward_prop(trainxmini, trainymini, weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)

                                    trajectory = np.vstack([trajectory,np.copy(weightsAndBiases)])

                            # val_cost calculation
                            val_cost, val_acc, val_zs, val_hs,val_yhat = forward_prop(valX, valY, weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)
                            #Calculate validation cost
                            print('validation cost',val_cost)
                            print('validation accuracy',val_acc)

                            #store optimal parameters if validation cost is less than optim mse 
                            if val_acc > optim_acc:
                                optim_acc = val_acc
                                optim_weightsAndBiases = weightsAndBiases
                                optim_lr = learn_rate
                                optim_epoch = num_epochs
                                optim_alpha = alpha
                                optim_batch = batch_size
                                optim_hidden_layers = NUM_HIDDEN_LAYERS
                                optim_num_hidden  = NUM_HIDDEN

    # print(optim_weightsAndBiases)
    test_cost,test_acc, test_zs, test_hs, test_yhat = forward_prop(xtest, ytest, optim_weightsAndBiases, NUM_HIDDEN_LAYERS, NUM_HIDDEN)
    print('test_cost', test_cost )
    print('test_Accuracy',test_acc)

    #Print Optimal hyperparameters and Test CE and accuracy
    print('Optimal alpha:', optim_alpha)
    print('Optimal batch size:', optim_batch)
    print('Optimal number of epochs:', optim_epoch)
    print('Optimal learning rate:', optim_lr)
    print('Optimal num hidden layers:', NUM_HIDDEN_LAYERS)
    print('Optimal num hidden:', NUM_HIDDEN)

    return trajectory

if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    #Loading Data
    # 60000, 78                          4
    X = np.load("fashion_mnist_train_images.npy")/255.0
    # 60000, 
    Y = np.load("fashion_mnist_train_labels.npy")
    # print(Y)
    # 10000, 784
    Xte = np.load("fashion_mnist_test_images.npy")/255.0
    # 10000, 
    Yte = np.load("fashion_mnist_test_labels.npy")

    # One hot encoding
    num_classes = 10
    Yhot = onehotencoder(Y,num_classes)
    Yhot_test = onehotencoder(Yte,num_classes)

    #trainX reshaped to 784,48000; trainY reshaped to 10,48000
    trainX = X.T
    trainY = Yhot.T
    testX = Xte.T
    testY = Yhot_test.T

    # Initialize weights and biases randomly 
    # weightsandbiases shape - 8180,
    weightsAndBiases = initWeightsAndBiases(NUM_HIDDEN_LAYERS, NUM_HIDDEN)

    # Perform gradient check on 5 training examples
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab, NUM_HIDDEN_LAYERS, NUM_HIDDEN)[0], \
                                    lambda wab: back_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab, NUM_HIDDEN_LAYERS, NUM_HIDDEN), \
                                    weightsAndBiases))


    # trajectory = findBestHyperparameters(trainX, trainY, testX, testY)
    
    _, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)

    # Plot the SGD trajectory
    plotSGDPath(trainX[:,:200], trainY[:,:200], trajectory)
