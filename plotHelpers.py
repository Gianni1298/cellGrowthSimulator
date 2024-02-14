import os

import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import voronoi_plot_2d


def save_plot(fig, filename):
    plt.savefig(f'output_plots/{filename}.png')
    plt.close(fig)


def create_gif(gif_name):
    path = 'output_plots'

    filenames = [f for f in os.listdir(path) if f.endswith('.png')]
    # Sorting files numerically based on the number in the filename
    filenames.sort(key=lambda x: int(x.split('.')[0]))

    images = []
    for filename in filenames:
        images.append(imageio.v3.imread(os.path.join(path, filename)))

    # Check if file exists and append a number starting from 1
    base_name = gif_name
    counter = 1
    gif_path = os.path.join(path, f'gif/gif_{base_name}_{counter}.gif')
    while os.path.exists(gif_path):
        counter += 1
        gif_path = os.path.join(path, f'gif/gif_{base_name}_{counter}.gif')

    imageio.mimsave(gif_path, images, duration=0.5)

    # Save the last frame separately
    last_frame = images[-1]
    last_frame_path = os.path.join(path, 'last_frames', f'last_frame_{base_name}_{counter}.png')
    imageio.v3.imwrite(last_frame_path, last_frame)


    for filename in filenames:
        os.remove(os.path.join(path, filename))


def createCDFPlot(areas, string_params):
    plt.figure()
    areas_sorted = np.sort(areas)
    cum_probs = np.arange(1, len(areas) + 1) / len(areas)
    plt.plot(areas_sorted, cum_probs)
    plt.xlabel('Area')
    plt.xlim(0, areas_sorted[-1])
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function of Cell Areas')

    # Increment filename if file already exists
    base_name = 'CDF'
    counter = 1
    path = 'output_plots/cdf'
    file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')

    plt.savefig(file_path)
    plt.close()

def createVoronoiPlot(vor, grid_bounds, areas, string_params):
    xmin, xmax, ymin, ymax = grid_bounds
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # First subplot: Custom Voronoi plot within grid bounds
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region] + [vor.vertices[region[0]]]
            if all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in polygon):
                ax1.fill(*zip(*polygon), alpha=0.4) # Fill the polygon
                ax1.plot(*zip(*polygon), color='black', markersize=0.8)  # Plot the edges

    # Plot the centers of the Voronoi regions
    ax1.plot(vor.points[:, 0], vor.points[:, 1], 'ko', markersize=3)  # 'ko' for black dots

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title('Filtered Voronoi Diagram')

    # Second subplot: Standard Voronoi plot
    voronoi_plot_2d(vor, ax=ax2)
    ax2.set_title('Standard Voronoi Diagram')

    # Third subplot: CDF
    areas_sorted = np.sort(areas)
    cum_probs = np.arange(1, len(areas) + 1) / len(areas)
    ax3.plot(areas_sorted, cum_probs)
    ax3.set_xlabel('Area')
    ax3.set_xlim(0, 40)
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function of Cell Areas')

    # Layout adjustments
    plt.tight_layout()


    # Increment filename if file already exists
    base_name = 'voronoi'
    counter = 1
    path = 'output_plots/voronoi'
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
    path = 'output_plots/ft'
    file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')

    plt.savefig(file_path)
    plt.close()


def createRipleyPlots(points, nearest_neighbor_distances, ripleyG, ripleyF, ripleyJ, ripleyK, ripleyL, string_params):
    # Create a figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # First subplot: Nearest neighbor distances
    axes[0, 0].scatter(points[:, 0], points[:, 1], label='Cells')
    for point, distance in zip(points, nearest_neighbor_distances):
        circle = plt.Circle(point, distance, color='r', fill=False, linestyle='--', linewidth=0.5)
        axes[0, 0].add_patch(circle)
    axes[0, 0].set_title('Nearest Neighbor Analysis')
    axes[0, 0].legend()

    # Second subplot: Ripley's G function
    G_middle_95pct = np.percentile(ripleyG.simulations, q=(2.5, 97.5), axis=0) # grab the middle 95% of simulations using numpy
    # use the fill_between function to color between the 2.5% and 97.5% envelope
    axes[0, 1].fill_between(ripleyG.support, *G_middle_95pct, color='lightgrey', label='simulated')

    # plot the line for the observed value of G(d)
    axes[0, 1].plot(ripleyG.support, ripleyG.statistic,
             color='orangered', label='observed')
    # and plot the support points depending on whether their p-value is smaller than .05
    axes[0, 1].scatter(ripleyG.support, ripleyG.statistic,
                cmap='viridis', c=ripleyG.pvalue < .01)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Distance')
    axes[0, 1].set_ylabel('G Function')
    axes[0, 1].set_title('G Function Plot')

    # Third subplot: Ripley's F function
    F_middle_95pct = np.percentile(ripleyF.simulations, q=(2.5, 97.5), axis=0) # grab the middle 95% of simulations using numpy
    # use the fill_between function to color between the 2.5% and 97.5% envelope
    axes[0, 2].fill_between(ripleyF.support, *F_middle_95pct, color='lightgrey', label='simulated')

    # plot the line for the observed value of F(d)
    axes[0, 2].plot(ripleyF.support, ripleyF.statistic,
             color='orangered', label='observed')
    # and plot the support points depending on whether their p-value is smaller than .05
    axes[0, 2].scatter(ripleyF.support, ripleyF.statistic,
                cmap='viridis', c=ripleyF.pvalue < .01)
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('Distance')
    axes[0, 2].set_ylabel('F Function')
    axes[0, 2].set_title('F Function Plot')

    # Fourth subplot: Ripley's J function
    # plot the line for the observed value of J(d)
    axes[1, 0].plot(ripleyJ.support, ripleyJ.statistic, color='orangered', label='observed')
    axes[1, 0].axhline(1, linestyle=':', color='k', label='Complete Spatial Randomness')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Distance')
    axes[1, 0].set_ylabel('J Function')
    axes[1, 0].set_title('J Function Plot')

    # Fifth subplot: Ripley's K function
    K_middle_95pct = np.percentile(ripleyK.simulations, q=(2.5, 97.5), axis=0) # grab the middle 95% of simulations using numpy
    # use the fill_between function to color between the 2.5% and 97.5% envelope
    axes[1, 1].fill_between(ripleyK.support, *K_middle_95pct, color='lightgrey', label='simulated')

    # plot the line for the observed value of K(d)
    axes[1, 1].plot(ripleyK.support, ripleyK.statistic,
             color='orangered', label='observed')
    # and plot the support points depending on whether their p-value is smaller than .05
    axes[1, 1].scatter(ripleyK.support, ripleyK.statistic,
                cmap='viridis', c=ripleyK.pvalue < .01)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Distance')
    axes[1, 1].set_ylabel('K Function')
    axes[1, 1].set_title('K Function Plot')

    # Sixth subplot: Ripley's L function
    L_middle_95pct = np.percentile(ripleyL.simulations, q=(2.5, 97.5), axis=0) # grab the middle 95% of simulations using numpy
    # use the fill_between function to color between the 2.5% and 97.5% envelope
    axes[1, 2].fill_between(ripleyL.support, *L_middle_95pct, color='lightgrey', label='simulated')

    # plot the line for the observed value of L(d)
    axes[1, 2].plot(ripleyL.support, ripleyL.statistic,
             color='orangered', label='observed')
    # and plot the support points depending on whether their p-value is smaller than .05
    axes[1, 2].scatter(ripleyL.support, ripleyL.statistic,
                cmap='viridis', c=ripleyL.pvalue < .01)
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Distance')
    axes[1, 2].set_ylabel('L Function')
    axes[1, 2].set_title('L Function Plot')

    # Layout adjustments
    plt.tight_layout()

    # Increment filename if file already exists
    base_name = 'ripley'
    counter = 1
    path = 'output_plots/nna'
    file_path = os.path.join(path, f'{base_name}_{string_params}_{counter}.png')
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(path, f'{base_name}_{counter}.png')

    plt.savefig(file_path)

    plt.close()
