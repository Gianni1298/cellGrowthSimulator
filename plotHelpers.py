import os

import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import voronoi_plot_2d


def save_plot(fig, filename):
    plt.savefig(f'output_plots/v4/{filename}.png')
    plt.close(fig)


def create_gif(gif_name):
    path = 'output_plots/v4'

    filenames = [f for f in os.listdir(path) if f.endswith('.png')]
    # Sorting files numerically based on the number in the filename
    filenames.sort(key=lambda x: int(x.split('.')[0]))

    images = []
    for filename in filenames:
        images.append(imageio.v3.imread(os.path.join(path, filename)))

    # Check if file exists and append a number starting from 1
    base_name = gif_name
    counter = 1
    gif_path = os.path.join(path, f'{base_name}_{counter}.gif')
    while os.path.exists(gif_path):
        counter += 1
        gif_path = os.path.join(path, f'{base_name}_{counter}.gif')

    imageio.mimsave(gif_path, images, duration=0.5)

    for filename in filenames:
        os.remove(os.path.join(path, filename))


def createCDFPlot(areas, string_params):
    plt.figure()
    areas_sorted = np.sort(areas)
    cum_probs = np.arange(1, len(areas) + 1) / len(areas)
    plt.plot(areas_sorted, cum_probs)
    plt.xlabel('Area')
    plt.xlim(0, 100)
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function of Cell Areas')

    # Increment filename if file already exists
    base_name = 'CDF'
    counter = 1
    path = 'output_plots/v4/cdf'
    file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')

    plt.savefig(file_path)
    plt.close()

def createVoronoiPlot(vor, string_params):
    plt.figure()
    voronoi_plot_2d(vor)
    plt.title('Voronoi Diagram')

    # Increment filename if file already exists
    base_name = 'voronoi'
    counter = 1
    path = 'output_plots/v4/voronoi'
    file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')

    plt.savefig(file_path)
    plt.close()


def createFTPlot(frequency, total_magnitude, string_params):
    plt.figure()
    plt.stem(frequency, total_magnitude, 'b', markerfmt=" ", basefmt=" ")
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform of Cell Density')
    plt.grid(True)

    # Increment filename if file already exists
    base_name = 'FT'
    counter = 1
    path = 'output_plots/v4/ft'
    file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')

    plt.savefig(file_path)
    plt.close()