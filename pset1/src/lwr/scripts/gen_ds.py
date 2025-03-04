import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Scalar multiplier for Gaussian noise
EPS = 1e-1
DATA_DIR = '../'
X_MIN = -5
X_MAX = 5


def generate_sin(num_examples):
    """Generate dataset with examples drawn from y=sin(x) + noise."""
    examples = []
    for _ in range(num_examples):
        x_val = X_MIN + (X_MAX - X_MIN) * np.random.random()
        x_dict = {f'x_1': x_val,
                  'y': np.sin(x_val ** 2) + EPS * np.random.normal()}
        examples.append(x_dict)

    df = pd.DataFrame(examples)

    return df


def plot_dataset(df, output_path):
    """Plot a 2D dataset and write to output_path."""
    xs = np.array([[row['x_1']] for _, row in df.iterrows()])
    ys = np.array([row['y'] for _, row in df.iterrows()])

    plt.figure(figsize=(12, 8))

    # Plot data
    for x, y in zip(xs[:, 0], ys):
        plt.scatter(x, y, marker='x', c='blue', alpha=.5)

    # Plot sine wave
    t = np.arange(0.0, 2 * np.pi, 0.01)
    plt.plot(t, np.sin(t), color='red')

    plt.savefig(output_path)


if __name__ == '__main__':
    np.random.seed(229)

    for split, n in [('train', 300), ('valid', 200), ('test', 201)]:
        sin_df = generate_sin(num_examples=n)
        sin_df.to_csv(os.path.join(DATA_DIR, f'{split}.csv'), index=False)
        plot_dataset(sin_df, os.path.join(DATA_DIR, f'plot_{split}.eps'))
