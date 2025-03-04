import numpy as np
import util

# Dimension of x
d = 500
# List for lambda to plot
reg_list = [0, 1, 5, 10, 50, 250, 500, 1000]
# List of dataset sizes
n_list = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

def regression(train_path, validation_path):
    """Part (b): Double descent for unregularized linear regression.
    For a specific training set, obtain beta_hat and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: Validation error
    """
    x_train, y_train = util.load_dataset(train_path)
    x_validation, y_validation = util.load_dataset(validation_path)

    val_err = 0
    # *** START CODE HERE ***
    beta_hat = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train
    m = x_validation.shape[0]
    val_err = (((x_validation@beta_hat-y_validation).T)@(x_validation@beta_hat-y_validation))/(2*m)
    # *** END CODE HERE
    return val_err

def ridge_regression(train_path, validation_path):
    """Part (c): Double descent for regularized linear regression.
    For a specific training set, obtain beta_hat under different l2 regularization strengths
    and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: List of validation errors for different scaling factors of lambda in reg_list.
    """
    x_train, y_train = util.load_dataset(train_path)
    x_validation, y_validation = util.load_dataset(validation_path)

    val_err = []
    # *** START CODE HERE ***

    n,d = x_train.shape
    m = x_validation.shape[0]
    lamda_list = [0,1,5,10,50,250,500,1000]

    for lamda in lamda_list :
        beta_hat = np.linalg.pinv(x_train.T @ x_train + lamda*np.eye(d) )@x_train.T@y_train
        err =  (((x_validation@beta_hat-y_validation).T)@(x_validation@beta_hat-y_validation))/(2*m)
        val_err.append( err )

    # *** END CODE HERE
    return val_err

if __name__ == '__main__':
    val_err = []
    for n in n_list:
        val_err.append(regression(train_path='train%d.csv' % n, validation_path='validation.csv'))
    util.plot(val_err, 'unreg.png', n_list)

    val_errs = []
    for n in n_list:
        val_errs.append(ridge_regression(train_path='train%d.csv' % n, validation_path='validation.csv'))
    val_errs = np.asarray(val_errs).T
    util.plot_all(val_errs, 'reg.png', n_list)
