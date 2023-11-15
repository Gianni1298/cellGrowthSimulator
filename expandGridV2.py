# Adjusting the function to ensure blue hexagons get further apart radially from the center as the grid expands.

import os
import imageio

output_folder = 'output_plots'  # Name of the output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  # Create the output folder if it doesn't exist

filenames = []  # to store the paths of the files with the plots

def expand_and_color_grid_radially(target_size, blue_hex_count, filename_prefix):
    """
    Function to iteratively expand a hexagonal grid and color a fixed number of hexagons blue,
    moving them radially outward from the center as the grid size increases.

    :param target_size: The target size for both nx and ny.
    :param blue_hex_count: The fixed number of blue hexagons.
    """
    # Import the necessary modules (assuming they are available in the user's environment)
    import matplotlib.pyplot as plt
    from hexalattice.hexalattice import create_hex_grid, plot_single_lattice_custom_colors

    # Initial size
    nx, ny = 20, 20

    # Loop through each iteration, expanding the grid each time
    while nx <= target_size and ny <= target_size:
        # Create the hexagonal grid for the current size
        hex_centers, _ = create_hex_grid(nx=nx, ny=ny, min_diam=1, do_plot=False)

        # Calculate the radial distances from the center of the grid
        center_x = hex_centers[:, 0].mean()
        center_y = hex_centers[:, 1].mean()
        distances = ((hex_centers[:, 0] - center_x)**2 + (hex_centers[:, 1] - center_y)**2)**0.5

        # Sort hexagons by distance from the center
        sorted_indices = distances.argsort()
        sorted_distances = distances[sorted_indices]

        # Select indices for blue hexagons such that they are spread out radially
        step = len(sorted_distances) // blue_hex_count
        blue_indices = sorted_indices[::step][:blue_hex_count]

        # Color the selected hexagons blue and the rest white
        colors = ['b' if i in blue_indices else 'w' for i in range(len(hex_centers))]

        # Now plot the hexagonal grid with the specified colors
        fig, ax = plt.subplots()
        plot_single_lattice_custom_colors(hex_centers[:, 0], hex_centers[:, 1],
                                          face_color=colors,
                                          edge_color='k',  # Keep the edges black for visibility
                                          min_diam=0.9,
                                          plotting_gap=0,
                                          rotate_deg=0)
        plt.title(f'Hexagonal Grid Size {nx}x{ny}')
        # plt.show()

        # Increase the grid size for the next iteration
        nx += 1
        ny += 1

        # Save the figure with the output folder in the path
        filename = os.path.join(output_folder, f'{filename_prefix}_plot_{nx}_{ny}.png')
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()  # Close the figure to avoid memory issues



expand_and_color_grid_radially(target_size=50, blue_hex_count=300, filename_prefix="300Blue")

# Create the GIF and save it in the output folder
gif_path = os.path.join(output_folder, '300Blue.gif')
with imageio.get_writer(gif_path, mode='I', duration=0.05) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optionally, remove files after creating the GIF if they're no longer needed
for filename in filenames:
    os.remove(filename)