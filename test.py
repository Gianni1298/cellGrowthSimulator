from hexalattice.hexalattice import *

# Generate hexagonal grid
hex_centers, _ = create_hex_grid(nx=5, ny=5, min_diam=1, do_plot=True)
x_hex_coords = hex_centers[:, 0]
y_hex_coords = hex_centers[:, 1]

plt.show()

# Create a list of the color blue, one for each hexagon
colors = ['b'] * len(hex_centers)

# Now plot the hexagonal grid with blue color
fig, ax = plt.subplots()
plot_single_lattice_custom_colors(x_hex_coords, y_hex_coords,
                                  face_color=colors,
                                  edge_color=colors,
                                  min_diam=0.9,
                                  plotting_gap=0,
                                  rotate_deg=0)

plt.show()

