import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def parzen_estimation(x_samples, point_x, h):
    k_n = 0
    for row in x_samples:
        x_i = (point_x - row[:, np.newaxis]) / h
        for row in x_i:
            if np.abs(row) > 0.5:
                break
        else:
            k_n += 1
    return (k_n / len(x_samples)) / (h ** point_x.shape[0])

def generate_data(n=10000):
    np.random.seed(123)
    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    samples = np.random.multivariate_normal(mu, cov, n)
    point = np.array([[0], [0]])
    widths = np.arange(0.1, 1.3, 0.1)
    return samples, point, widths

def save_data(widths, results, filename):
    df = pd.DataFrame({'h': widths, 'p(x)': results})
    df.to_csv(filename, index=False)
    print(f"Resultados guardados en {filename}")

def save_benchmarks(benchmarks, filename):
    df = pd.DataFrame({
        'method': ['serial', '2', '3', '4', '6'],
        'time_seconds': benchmarks
    })
    df.to_csv(filename, index=False)
    print(f"Tiempos guardados en {filename}")

def plot_results(widths, results, reference_time=None, title='', filename=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.barh(range(len(results)), results, color='#8fd19e', edgecolor='black')
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f'{w:.1f}' for w in widths])
    ax.invert_yaxis()
    ax.set_xlabel('Estimated Density')
    ax.set_ylabel('h (window width)')
    ax.set_title(title)

    if reference_time:
        ax.axvline(np.mean(results), color='gray', linestyle='dashed')

    for bar, val in zip(bars, results):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va='center', ha='left', fontsize=10)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"Gráfico guardado en {filename}")
    plt.show()

def plot_benchmark_results(benchmarks, filename=None):
    bar_labels = ['serial', '2', '3', '4', '6']
    y_pos = np.arange(len(benchmarks))
    serial_time = benchmarks[0]

    fig, ax = plt.subplots(figsize=(7, 5))  # más ancho, menos alto

    bars = ax.barh(
        y=y_pos,
        width=benchmarks,
        color='#a1d99b',
        edgecolor='black',
        height=0.6  # barras más gruesas
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel('time in seconds for n=10000', fontsize=12)
    ax.set_ylabel('number of processes', fontsize=12)
    ax.set_title('Serial vs. Multiprocessing via Parzen-window estimation', fontsize=14, pad=10)

    for i, (bar, time) in enumerate(zip(bars, benchmarks)):
        percent = (serial_time / time) * 100
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{percent:.2f}%",
            va='center',
            ha='left',
            fontsize=11
        )

    ax.axvline(serial_time, color='gray', linestyle='dashed', linewidth=1)
    ax.set_xlim(0, max(benchmarks) * 1.15)
    ax.set_ylim(-0.5, len(benchmarks) - 0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"Gráfico guardado en: {filename}")
    plt.show()

if __name__ == '__main__':
    
    # ====== Generacion de ejemplo
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("auto")

    # Plot Points

    # samples within the cube
    X_inside = np.array([[0,0,0],[0.2,0.2,0.2],[0.1, -0.1, -0.3]])

    X_outside = np.array([[-1.2,0.3,-0.3],[0.8,-0.82,-0.9],[1, 0.6, -0.7],
                        [0.8,0.7,0.2],[0.7,-0.8,-0.45],[-0.3, 0.6, 0.9],
                        [0.7,-0.6,-0.8]])

    for row in X_inside:
        ax.scatter(row[0], row[1], row[2], color="r", s=50, marker='^')

    for row in X_outside:    
        ax.scatter(row[0], row[1], row[2], color="k", s=50)

    # Plot Cube
    h = [-0.5, 0.5]
    for s, e in combinations(np.array(list(product(h,h,h))), 2):
        if np.sum(np.abs(s-e)) == h[1]-h[0]:
            ax.plot3D(*zip(s,e), color="g")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    plt.savefig('../plots/example.png')
