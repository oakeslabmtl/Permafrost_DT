import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# ------------------------------------------------------------
# 1. FEM domain and mesh
# ------------------------------------------------------------

Lx = 24.0      # cm
Lz = 30.0      # cm

dx = 1.0       # cm
dz = 1.0       # cm

x_vec = np.arange(0.0, Lx + dx, dx)
z_vec = np.arange(0.0, Lz + dz, dz)

nx = len(x_vec)   # number of nodes in x direction = 25
nz = len(z_vec)   # number of nodes in z direction = 31

num_nodes = nx * nz


# ------------------------------------------------------------
# 2. Generate FEM nodes row by row
# Node numbering starts from bottom-left and goes to right
# ------------------------------------------------------------

node_id = np.zeros(num_nodes, dtype=int)
x_fem = np.zeros(num_nodes)
z_fem = np.zeros(num_nodes)

counter = 0

for j in range(nz):          # bottom to top
    for i in range(nx):      # left to right

        node_id[counter] = counter + 1

        x_fem[counter] = x_vec[i]
        z_fem[counter] = z_vec[j]

        counter += 1


# Node table:
# Column 1: node number
# Column 2: x coordinate
# Column 3: z coordinate
nodes = np.column_stack((node_id, x_fem, z_fem))


# ------------------------------------------------------------
# 3. Sensor coordinates
# ------------------------------------------------------------

# Height from bottom [cm]
height_sensor = np.array([29, 25, 21, 17, 13, 9, 5, 1], dtype=float)

# Depth from left side [cm]
depth_sensor = np.array([0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 13.5, 16.5], dtype=float)

# Create sensor coordinate grid
x_sensor_grid, z_sensor_grid = np.meshgrid(depth_sensor, height_sensor)


# ------------------------------------------------------------
# 4. Experimental temperature data
# ------------------------------------------------------------
# T_sensor must be 8 x 10.
# Rows correspond to height_sensor.
# Columns correspond to depth_sensor.

T_sensor = np.array([
    # depth: 0      1.5     3       4.5     6       7.5     9       10.5    13.5    16.5
    [-2.962, -2.869, -2.700, -2.857, -3.303, -3.775, -3.276, -3.486, -3.171, -2.831],  # height = 29
    [-3.801, -4.313, -4.353, -4.353, -4.353, -4.353, -4.379, -4.327, -4.300, -4.248],  # height = 25
    [-4.669, -4.918, -4.984, -4.984, -4.984, -4.958, -4.906, -5.050, -5.076, -5.023],  # height = 21
    [-5.076, -5.287, -5.471, -5.445, -5.366, -5.313, -5.287, -5.260, -5.313, -5.287],  # height = 17
    [-5.577, -5.841, -5.841, -5.841, -5.788, -5.735, -5.682, -5.656, -5.524, -5.761],  # height = 13
    [-5.788, -5.950, -6.104, -6.078, -6.052, -6.025, -5.999, -5.946, -5.920, -6.025],  # height = 9
    [-5.867, -5.924, -5.999, -5.946, -6.369, -6.342, -6.342, -6.316, -6.236, -6.184],  # height = 5
    [-6.263, -6.263, -6.263, -6.527, -6.369, -6.417, -6.554, -6.322, -6.131, -6.448]   # height = 1
])


# ------------------------------------------------------------
# 5. Convert sensor data to scattered data
# ------------------------------------------------------------

x_sensor = x_sensor_grid.ravel()
z_sensor = z_sensor_grid.ravel()
T_sensor_vec = T_sensor.ravel()

sensor_points = np.column_stack((x_sensor, z_sensor))


# ------------------------------------------------------------
# 6. Interpolate temperature to FEM nodes
# ------------------------------------------------------------

fem_points = np.column_stack((x_fem, z_fem))

# Interpolation inside sensor convex hull
linear_interpolator = LinearNDInterpolator(sensor_points, T_sensor_vec)

# Extrapolation outside sensor convex hull using nearest sensor
nearest_interpolator = NearestNDInterpolator(sensor_points, T_sensor_vec)

T_initial_nodes = linear_interpolator(fem_points)

# Replace NaN values outside interpolation region with nearest extrapolation
nan_mask = np.isnan(T_initial_nodes)
T_initial_nodes[nan_mask] = nearest_interpolator(fem_points[nan_mask])


# ------------------------------------------------------------
# 7. Final FEM initial condition table
# ------------------------------------------------------------

# Column 1: node number
# Column 2: x coordinate
# Column 3: z coordinate
# Column 4: initial temperature
initial_temperature_table = np.column_stack(
    (node_id, x_fem, z_fem, T_initial_nodes)
)


# ------------------------------------------------------------
# 8. Reshape temperature to grid for plotting
# ------------------------------------------------------------
# Since node numbering is row by row:
# row 1 = z = 0
# row 2 = z = 1
# ...
# row 31 = z = 30

T0_grid = T_initial_nodes.reshape((nz, nx))
x_grid = x_fem.reshape((nz, nx))
z_grid = z_fem.reshape((nz, nx))


# ------------------------------------------------------------
# 9. Plot interpolated temperature field
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

contour = plt.contourf(
    x_grid,
    z_grid,
    T0_grid,
    levels=30
)

plt.colorbar(contour, label="Temperature [°C]")

plt.scatter(
    x_sensor,
    z_sensor,
    s=80,
    c=T_sensor_vec,
    edgecolors="k"
)

plt.xlabel("Depth from left side, x [cm]")
plt.ylabel("Height from bottom, z [cm]")
plt.title("Interpolated Initial Temperature Field")
plt.axis("equal")
plt.xlim(0, Lx)
plt.ylim(0, Lz)
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. Save data
# ------------------------------------------------------------

header = "node_id,x_cm,z_cm,initial_temperature_C"

np.savetxt(
    "initial_temperature_nodes.csv",
    initial_temperature_table,
    delimiter=",",
    header=header,
    comments="",
    fmt=["%d", "%.6f", "%.6f", "%.6f"]
)

np.savez(
    "initial_temperature_nodes.npz",
    initial_temperature_table=initial_temperature_table,
    node_id=node_id,
    x_fem=x_fem,
    z_fem=z_fem,
    T_initial_nodes=T_initial_nodes,
    T0_grid=T0_grid,
    x_grid=x_grid,
    z_grid=z_grid
)

print("Data saved:")
print("initial_temperature_nodes.csv")
print("initial_temperature_nodes.npz")