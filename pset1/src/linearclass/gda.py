import numpy as np
import util





class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples , dim = x.shape

        # CALCULATING THE TERMS REQUREID USING PROB Q4 D)
        phi = np.sum( y[y==1] ) / n_examples

        mu0= np.mean( x[y==0] , axis=0 )
        mu1= np.mean( x[y==1] , axis=0 )

        term1 =x[y==0] - mu0
        term2 = x[y==1] - mu1

        sigma= 1/n_examples *( np.matmul(term1.T,term1) + np.matmul(term2.T,term2))
        theta = np.zeros(3)
        theta[0] =-1/2*(np.matmul(mu1.T,np.matmul(np.linalg.inv(sigma) ,mu1))-(np.matmul(mu0.T,np.matmul(np.linalg.inv(sigma) ,mu0))))-np.log((1-phi)/phi)
        theta[1:]= np.matmul(np.linalg.inv(sigma) , (mu1-mu0) )
        self.theta = theta
        # *** END CODE HERE ***
        
        if self.verbose:
            print('Final theta (GDA): {}'.format(self.theta))

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 +np.exp(-x @ self.theta ))
        # *** END CODE HERE


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    clf = None

    # *** START CODE HERE ***
    clf = GDA(theta_0 = np.zeros(3))
    clf.fit(x_train,y_train)
    # *** END CODE HERE ***
    
    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('LR Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')


