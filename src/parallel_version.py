from utils import parzen_estimation, generate_data, plot_results, save_data, plot_benchmark_results, save_benchmarks
import os
import time
import multiprocessing as mp

os.makedirs('../plots', exist_ok=True)
os.makedirs('../data', exist_ok=True)

def multiprocess(samples, point, widths, processes):
    with mp.Pool(processes=processes) as pool:
        results = pool.starmap(parzen_estimation, [(samples, point, w) for w in widths])
    return results

samples, point, widths = generate_data()

# Benchmark test
benchmarks = []

# Serial
start = time.time()
[parzen_estimation(samples, point, w) for w in widths]
benchmarks.append(time.time() - start)

# Paralelo con distintos números de procesos
for p in [2, 3, 4, 6]:
    start = time.time()
    multiprocess(samples, point, widths, processes=p)
    benchmarks.append(time.time() - start)

# Guardar resultados y graficar
save_benchmarks(benchmarks, '../data/benchmark_times.csv')
plot_benchmark_results(benchmarks, filename='../plots/benchmark_plot.png')
