import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression

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


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    best_tau = None
    best_mse = None

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid,y_valid = util.load_dataset(valid_path,add_intercept=True)
    # x_eval (200,2) y_eval (200,)

    mse_values = [] 
    cnt = 1
    for tau in tau_values :
        model = LocallyWeightedLinearRegression(tau) # defining the model with a tau
        model.fit( x_train,y_train ) # setting the value of model's x and y .

        p_valid = model.predict( x_valid.copy() )  
        mse_values.append(float(np.mean((p_valid - y_valid) ** 2)))
        plot(x_valid[:, 1:], p_valid, x_train[:, 1], y_train, f'plot{cnt}.png')
        cnt += 1


    
    best_mse = min(mse_values)
    print( mse_values )
    min_idx = mse_values.index(best_mse)
    best_tau = tau_values[min_idx]
        
      # *** END CODE HERE ***

    # Fit a LWR model with the best tau value
    print('Best tau: {} (MSE: {:g})'.format(best_tau, best_mse))
    clf = LocallyWeightedLinearRegression(best_tau)
    clf.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    p_test = clf.predict(x_test)
    mse = np.mean((p_test - y_test) ** 2)
    print('Test MSE: {:g}'.format(mse))
    # Save predictions to pred_path
    np.savetxt(pred_path, p_test)




if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
