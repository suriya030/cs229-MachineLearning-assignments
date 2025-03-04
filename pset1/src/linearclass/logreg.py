import numpy as np
import util

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
    
    def sigmoid(self,z):
        return 1 / (1+np.exp(-z))
    
    def loss(self,x,y):
        h = self.sigmoid(np.matmul(x,self.theta))
        return -np.mean( y *np.log( 1e-10+h) + (1-y)*np.log(1e-10+1-h) )

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape
        self.theta = np.zeros(dim)

        for i in range(0,self.max_iter):
            h = self.sigmoid(np.matmul(x,self.theta)) # predicting probablities
            # h (n_examples)
            
            grad = np.matmul(x.T,(h-y))/n_examples
            hessian = np.matmul(x.T,np.matmul(np.diag(h*(1-h)),x)) / n_examples 

            det = np.linalg.det(hessian)
            if  det == 0 :
                print( "Hessian is not invertible !")
                break
            change = np.matmul(np.linalg.inv(hessian),grad)        
            self.theta = self.theta-change

            loss = self.loss(x,y)
            if self.verbose :
                print( "Loss value :" , self.loss(x,y) )

            if( np.linalg.norm(change,ord=1)) < self.eps :
                print("Converged in interations : ",i+1)
                break
        
        # *** END CODE HERE ***
        if self.verbose:
            print('Final theta (logreg): {}'.format(self.theta))

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(np.matmul(x,self.theta))
        # *** END CODE HERE ***




def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # for ds1_train , x_train (800,3) y_train (800,)
    # for ds2_train , x_train (800,3) y_train (800,)
    clf = None

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit( x_train , y_train ) # now we have got optimum theta here
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
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')