def expand_and_color_grid(target_size, blue_hex_count):
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
    nx, ny = 5, 5

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

        # Color the first 'blue_hex_count' hexagons blue, and the rest white
        colors = ['b' if i < blue_hex_count else 'w' for i in range(len(hex_centers))]

        # Sort the colors array according to the sorted indices
        sorted_colors = [colors[i] for i in sorted_indices]

        # Now plot the hexagonal grid with the specified colors
        fig, ax = plt.subplots()
        plot_single_lattice_custom_colors(hex_centers[:, 0], hex_centers[:, 1],
                                          face_color=sorted_colors,
                                          edge_color='k',  # Keep the edges black for visibility
                                          min_diam=0.9,
                                          plotting_gap=0,
                                          rotate_deg=0)
        plt.title(f'Hexagonal Grid Size {nx}x{ny}')
        plt.show()

        # Increase the grid size for the next iteration
        nx += 1
        ny += 1


expand_and_color_grid(target_size=20, blue_hex_count=30)

