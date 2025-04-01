from utils import parzen_estimation, generate_data, plot_results, save_data
import os
import time

os.makedirs('../plots', exist_ok=True)
os.makedirs('../data', exist_ok=True)

samples, point, widths = generate_data()
start = time.time()
results = [parzen_estimation(samples, point, w) for w in widths]
end = time.time()

time_taken = end - start
save_data(widths, results, filename='../data/serial_results.csv')
plot_results(widths, results, reference_time=time_taken, title='Serial Estimation', filename='../plots/serial_plot.png')

print(f"Tiempo total (serial): {time_taken:.4f} segundos")