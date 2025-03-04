import numpy as np
import matplotlib.pyplot as plt
import argparse

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    n_examples , k = x.shape
    maxx = np.max(x,axis=1,keepdims=True) # (n_examples,1) 
    softmax = np.exp(x-maxx) / np.sum(np.exp(x-maxx),axis=1,keepdims=True)   

    return softmax # (n_examples,k)
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    return 1.0 / (1.0 + np.exp(-x)) # (n_examples,dim)
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    d = input_size 
    m = num_hidden
    k = num_output

    W1 = np.random.normal(loc=0.0,scale=1.0,size=(d,m))
    b1 = np.zeros(m)
    W2 = np.random.normal(loc=0.0,scale=1.0,size=(m,k))
    b2 = np.zeros(k)

    dict = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
   }
    
    return dict
    # *** END CODE HERE ***

def forward_prop(data, one_hot_labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    
    # d = 784 , m = 300 , k = 10 
    # data (n_examples,d) , one_hot_labels (n_examples,k)
    n_examples,d = data.shape 
   
    X = data # (n_examples,d)
    W1 = params['W1'] # (d,m)
    b1 = params['b1'] # (m,)
    W2 = params['W2'] # (m,k)
    b2 = params['b2'] # (k,)

    Z1 = np.matmul(X,W1)+b1.T # (n_examples,m) 
    a1 = sigmoid( Z1 ) # (n_examples,m)

    Z2 = np.matmul(a1,W2)+b2 # (n_examples,k) Z2 aka logits
    output = softmax(Z2) # (n_examples,k)

    # Loss calculation  
    # one_hot_labels (n_examples,k)
    avg_loss = -1.0*np.mean(np.log(np.sum(one_hot_labels*output,axis=1)))

    return a1 , output , avg_loss 
    # *** END CODE HERE ***

def backward_prop(data, one_hot_labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    n_examples,d = data.shape
    # one_hot_labels (n_examples,k)
    a1 , h_theta , loss = forward_prop_func ( data , one_hot_labels , params )
    # a1:(n_examples,m) h_theta:(n_examples,k)

    gradJ_a2 = h_theta - one_hot_labels # (n_examples,k)
    gradJ_W2 = np.dot(a1.T,gradJ_a2) / n_examples 
    gradJ_b2 = np.sum(gradJ_a2,axis=0) / n_examples

    gradJ_a1 = np.dot(gradJ_a2,params['W2'].T) 
    gradJ_z1 = a1*(1-a1)*gradJ_a1 
    gradJ_W1 = np.dot(data.T,gradJ_z1) /n_examples 
    gradJ_b1 = np.sum(gradJ_z1,axis=0) / n_examples 

    return {'W1': gradJ_W1, 'b1': gradJ_b1, 'W2': gradJ_W2, 'b2': gradJ_b2}
    # *** END CODE HERE ***


def backward_prop_regularized(data, one_hot_labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input                                                
        one_hot_labels: A 2d numpy array containing the the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    n_examples,d = data.shape
    # one_hot_labels (n_examples,k)
    a1 , h_theta , loss = forward_prop_func ( data , one_hot_labels , params )
    # a1:(n_examples,m) h_theta:(n_examples,k)

    gradJ_a2 = h_theta - one_hot_labels # (n_examples,k)
    gradJ_W2 = np.dot(a1.T,gradJ_a2) / n_examples +2*(reg)*params['W2']
    gradJ_b2 = np.sum(gradJ_a2,axis=0) / n_examples

    gradJ_a1 = np.dot(gradJ_a2,params['W2'].T) 
    gradJ_z1 = a1*(1-a1)*gradJ_a1 
    gradJ_W1 = np.dot(data.T,gradJ_z1) /n_examples +2*(reg)*params['W1']
    gradJ_b1 = np.sum(gradJ_z1,axis=0) / n_examples 

    return {'W1': gradJ_W1, 'b1': gradJ_b1, 'W2': gradJ_W2, 'b2': gradJ_b2}
    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, one_hot_train_labels, learning_rate, batch_size, params,
                            forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        one_hot_train_labels: A numpy array containing the one-hot embeddings of the training labels e_y.
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    n_examples , d = train_data.shape
    n_iterations = n_examples//batch_size # assuming batch_size | n_examples
    for i in range(0,n_iterations):
        
        start_idx = i*batch_size
        end_idx = start_idx + batch_size # not inclusive
        data = train_data[start_idx:end_idx,:]
        one_hot_labels = one_hot_train_labels[start_idx:end_idx,:]

        # Computing the gradients
        grad = backward_prop_func(data,one_hot_labels,params,forward_prop_func)

        # Updating the parameters
        params['W1'] = params['W1'] - learning_rate*grad['W1']
        params['b1'] = params['b1'] - learning_rate*grad['b1']
        params['W2'] = params['W2'] - learning_rate*grad['W2']
        params['b2'] = params['b2'] - learning_rate*grad['b2']


    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    # train_data (60000,784) , train_label (60000,)
    # convert labels to one-hot embeddings e_y.
    train_labels = one_hot_labels(train_labels) # (60000,10)
    p = np.random.permutation(60000)
    # random shuffling the data
    train_data = train_data[p,:] 
    train_labels = train_labels[p,:]

    # splitting the data into train and dev.
    dev_data = train_data[0:10000,:]  # (10000,784)
    dev_labels = train_labels[0:10000,:] # (10000,)
    train_data = train_data[10000:,:] # (50000,784)
    train_labels = train_labels[10000:,:] # (50000,)

    # scaling the data
    mean = np.mean(train_data) # mean of the flattened 2D array
    std = np.std(train_data) # std of the flattened 2D array
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    # convert labels to one-hot embeddings e_y.
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    baseline_acc,reg_acc = -1,-1
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)
        
    return baseline_acc, reg_acc


if __name__ == '__main__':
    main()