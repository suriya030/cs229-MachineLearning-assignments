import matplotlib.pyplot as plt
import numpy as np
import util

class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x.copy()
        self.y = y.copy()
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        
        if self.x is None or self.y is None:
            raise RuntimeError('Must call fit before predict.')

        # *** START CODE HERE ***
        y = []
        for x_test in x:
            W = np.diag(np.exp(-1 * np.power((self.x[:,1]-x_test[1]),2) / (2*self.tau*self.tau) ) )
            
            # A(theta) = b needs to be solved
            A = np.matmul(np.matmul(self.x.T,W),self.x)
            b = np.matmul(np.matmul(self.x.T,W),self.y)  
            theta = np.linalg.solve(A,b)
            y.append(np.matmul(theta.T,x_test))
        y = np.array( y )
        return y

        # *** END CODE HERE ***

def plot(x_eval, p_eval, x_train, y_train, save_path):
    """plot the data"""

    # *** START CODE HERE ***
    # make sure to save your plot with the provided `save_path`
    plt.figure()
    plt.scatter(x_train, y_train, color='blue', marker='x', label='training_data')
    plt.scatter(x_eval, p_eval, color='red', marker='o', label='validation_predictions')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    plt.savefig(save_path)
    plt.close()
    # *** END CODE HERE ***


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # x_train (300,2) , y_train (300,)
    p_eval = None
    y_eval = None

    # *** START CODE HERE **
    model = LocallyWeightedLinearRegression(tau) # defining the model with a tau
    model.fit( x_train,y_train ) # setting the value of model's x and y .

    x_eval,y_eval = util.load_dataset(eval_path,add_intercept=True)
    # x_eval (200,2) y_eval (200,)
    p_eval = model.predict( x_eval.copy() )
    # *** END CODE HERE ***

    print('Validation MSE: {:g}'.format(np.mean((p_eval - y_eval) ** 2)))

    # Plot validation predictions on top of training set
    # No need to save predictions
    plot_path = './plot.pdf'
    plot(x_eval[:, 1:], p_eval, x_train[:, 1], y_train, plot_path)

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
