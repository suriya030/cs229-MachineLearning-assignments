import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y): 
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***  
        
        # Assuming the cost function to be
        # J(theta) = 1/2 * (X*theta-Y)^T(X*Theta-Y)     
        self.theta = np.linalg.solve(np.matmul(X.T,X),np.matmul(X.T,y))

        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        if k == 0 :
            return X[:,0].reshape(-1,1)

        mat = [X]
        for i in range(2,k+1):
            mat.append( X[:,1].reshape(-1,1) ** i )

        if k >= 2 : 
            X = np.hstack(mat)
        
        return X
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        X_sin = np.sin(X[:,1]).reshape(-1,1) # [n_examples,1]
        X_poly = self.create_poly( k,X ) # [n_examples,k+1]
        X = np.hstack([X_poly , X_sin]) # [n_examples,k+2]

        return X
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.matmul(X,self.theta)        
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True) 
    # train_x (n_examples,2) and train_y (n_examples,1)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    # plot_x (1000,2)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)


    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''


        plot_y = None

        # *** START CODE HERE ***
        linear_model = LinearModel( )
        train_x_hat = train_x.copy()
        plot_x_hat = plot_x.copy()

        # creating feature vector from attribute vector
        
        if sine :
            train_x_hat= linear_model.create_sin(k,train_x_hat)
        else :
            train_x_hat= linear_model.create_poly(k,train_x_hat)
        
        # finding optimum theta
        linear_model.fit(train_x_hat,train_y)

        # prediting the model on plot_x data

        if sine :
            plot_x_hat= linear_model.create_sin(k,plot_x_hat)  
        else :
            plot_x_hat= linear_model.create_poly(k,plot_x_hat)     

        plot_y = linear_model.predict(plot_x_hat)
        # *** END CODE HERE ***
        
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    run_exp(small_path,sine=False,ks=[5])
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
