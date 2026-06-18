"""
Multiscale Simulation of Electrical Resistivity in Sn-Bi Alloys

Pipeline:
1. KMC-based microstructure generation (Ising model)
2. Fickian diffusion for solute redistribution
3. Phenomenological scattering model (linearized Nordheim-type)
4. Finite difference resistor network (Laplace solver)
5. 2D → 3D correction using Bakker EMT

Author: Advait Karmarkar
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import time
from numba import njit
from scipy.ndimage import convolve

# ==========================================
# 1. PHYSICS & GRID PARAMETERS
# ==========================================
GRID_SIZE = 1000
VOL_FRAC_SN = 0.49

INTERFACE_PENALTY = 45.0  # phenomenological boundary resistance (Sn–Bi interface)

# Baseline resistivities (µΩ·cm)
RHO_SN_BASE = 18.0
RHO_BI_BASE = 129.0

# Equilibrium solubility limits (room temperature)
BASE_SOL_BI_IN_SN = 0.03
BASE_SOL_SN_IN_BI = 0.001

# Solute trapping limits (interface enrichment)
MAX_SEGREGATION_BI_IN_SN = 0.21
MAX_SEGREGATION_SN_IN_BI = 0.02

# Scattering coefficients (linearized Nordheim-type model)
K_BI_IN_SN = 20.0
K_SN_IN_BI = 10000.0  # strong scattering due to low carrier density in Bi

PHASE_BI = 0
PHASE_SN = 1

KMC_STEPS = 50
KT = 0.65
J = 1.0

# Master random seed. Seeds BOTH NumPy's interpreter-level RNG (initial melt and
# diffusion noise) and Numba's independent RNG (the KMC engine), so the entire
# pipeline is reproducible. Results are bit-identical only within a fixed
# NumPy/Numba version (pin versions via requirements.txt) but are deterministic
# run-to-run on a given install.
SEED = 42

print(f"Initializing {GRID_SIZE}x{GRID_SIZE} random melt...")
np.random.seed(SEED)

# Random liquid initialization (volume fraction based)
phase_matrix = np.where(
    np.random.rand(GRID_SIZE, GRID_SIZE) > (1 - VOL_FRAC_SN),
    PHASE_SN,
    PHASE_BI
).astype(np.int8)

# ==========================================
# 2. KMC MICROSTRUCTURE GENERATION
# ==========================================
@njit
def run_kmc_growth(matrix, steps, kT, J, seed):
    # Seed Numba's internal RNG from *inside* the JIT-compiled function.
    # numpy.random.seed() called in interpreter scope does NOT control Numba's
    # RNG, so without this line the KMC microstructure (and the final result)
    # would differ on every run despite the global seed above.
    np.random.seed(seed)

    rows, cols = matrix.shape
    N = rows * cols

    # Precompute Boltzmann factors
    exp_table = np.zeros(9)
    for dE in range(9):
        exp_table[dE] = np.exp(-dE / kT)

    for _ in range(steps):
        for _ in range(N):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)

            direction = np.random.randint(0, 4)
            nx, ny = x, y
            if direction == 0: nx = (x + 1) % rows
            elif direction == 1: nx = (x - 1) % rows
            elif direction == 2: ny = (y + 1) % cols
            elif direction == 3: ny = (y - 1) % cols

            current_state = matrix[x, y]
            new_state = matrix[nx, ny]

            if current_state == new_state:
                continue

            E_initial, E_final = 0, 0

            # Compute local interaction energies
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nnx = (x + dx) % rows
                    nny = (y + dy) % cols
                    neighbor_state = matrix[nnx, nny]

                    if neighbor_state != current_state:
                        E_initial += J
                    if neighbor_state != new_state:
                        E_final += J

            dE = E_final - E_initial

            # Metropolis criterion
            if dE <= 0 or np.random.rand() < exp_table[int(dE)]:
                matrix[x, y] = new_state

    return matrix

print(f"Running KMC grain growth ({KMC_STEPS} steps)...")
start_kmc = time.time()
phase_matrix = run_kmc_growth(phase_matrix, KMC_STEPS, KT, J, SEED)
print(f"KMC completed in {time.time() - start_kmc:.2f} s")

# ==========================================
# 3. DIFFUSION (FICK'S SECOND LAW)
# ==========================================
print("Solving diffusion (FTCS scheme)...")
start_diff = time.time()

bi_mask = (phase_matrix == PHASE_BI)
sn_mask = (phase_matrix == PHASE_SN)

# Initialize impurity concentration fields
c_sn_impurity = np.full((GRID_SIZE, GRID_SIZE), BASE_SOL_SN_IN_BI)
c_bi_impurity = np.full((GRID_SIZE, GRID_SIZE), BASE_SOL_BI_IN_SN)

# Discrete Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Identify phase boundaries using convolution
sn_edges = np.logical_and(sn_mask,
    convolve(sn_mask.astype(int), laplacian_kernel, mode='constant') < 0)

bi_edges = np.logical_and(bi_mask,
    convolve(bi_mask.astype(int), laplacian_kernel, mode='constant') < 0)

# Diffusion parameters (CFL stable)
D_bi, D_sn = 0.15, 0.10
dt = 0.1
diffusion_steps = 150

# Evolve impurity fields
for _ in range(diffusion_steps):
    nabla2_c_bi = convolve(c_bi_impurity, laplacian_kernel, mode='nearest')
    nabla2_c_sn = convolve(c_sn_impurity, laplacian_kernel, mode='nearest')

    c_bi_impurity[sn_mask] += (D_bi * nabla2_c_bi * dt)[sn_mask]
    c_sn_impurity[bi_mask] += (D_sn * nabla2_c_sn * dt)[bi_mask]

    # Dirichlet pinning at interfaces (solute trapping)
    c_bi_impurity[sn_edges] = MAX_SEGREGATION_BI_IN_SN
    c_sn_impurity[bi_edges] = MAX_SEGREGATION_SN_IN_BI

# Small noise to avoid artificial symmetry
c_sn_impurity[bi_mask] += np.random.normal(0, 0.0002, size=np.sum(bi_mask))
c_bi_impurity[sn_mask] += np.random.normal(0, 0.0005, size=np.sum(sn_mask))

print(f"Diffusion completed in {time.time() - start_diff:.2f} s")

# ==========================================
# 4. SCATTERING MODEL (LINEARIZED NORDHEIM-TYPE)
# ==========================================
print("Applying impurity scattering model...")

rho_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

# Excess impurity above equilibrium solubility
delta_c_bi_in_sn = np.maximum(0, c_bi_impurity[sn_mask] - BASE_SOL_BI_IN_SN)
delta_c_sn_in_bi = np.maximum(0, c_sn_impurity[bi_mask] - BASE_SOL_SN_IN_BI)

# Linearized scattering model
rho_map[sn_mask] = RHO_SN_BASE + (K_BI_IN_SN * delta_c_bi_in_sn)
rho_map[bi_mask] = RHO_BI_BASE + (K_SN_IN_BI * delta_c_sn_in_bi)

# ==========================================
# 5. TRANSPORT SOLVER (FINITE DIFFERENCE)
# ==========================================
print("Building sparse conductance matrix...")
start_build = time.time()

N = GRID_SIZE * GRID_SIZE

# Horizontal conductance
rho1_h, rho2_h = rho_map[:, :-1], rho_map[:, 1:]
p1_h, p2_h = phase_matrix[:, :-1], phase_matrix[:, 1:]
penalty_h = np.where(p1_h != p2_h, INTERFACE_PENALTY, 0.0)
g_h = 1.0 / ((rho1_h + rho2_h) / 2.0 + penalty_h)

# Vertical conductance
rho1_v, rho2_v = rho_map[:-1, :], rho_map[1:, :]
p1_v, p2_v = phase_matrix[:-1, :], phase_matrix[1:, :]
penalty_v = np.where(p1_v != p2_v, INTERFACE_PENALTY, 0.0)
g_v = 1.0 / ((rho1_v + rho2_v) / 2.0 + penalty_v)

# Build matrix diagonals
main_diag_2D = np.zeros((GRID_SIZE, GRID_SIZE))
main_diag_2D[:, :-1] += g_h
main_diag_2D[:, 1:] += g_h
main_diag_2D[:-1, :] += g_v
main_diag_2D[1:, :] += g_v

main_diag = main_diag_2D.flatten()

g_h_padded = np.zeros((GRID_SIZE, GRID_SIZE))
g_h_padded[:, :-1] = g_h

right_diag = -g_h_padded.flatten()[:-1]
bottom_diag = -g_v.flatten()

A = sp.diags([main_diag, right_diag, bottom_diag],
             [0, 1, GRID_SIZE], shape=(N, N), format='lil')

# Enforce symmetry
A = A + A.T - sp.diags(A.diagonal())
A = A.tolil()

# Boundary conditions (1V → 0V)
b = np.zeros(N)

for y in range(GRID_SIZE):
    left = y * GRID_SIZE
    right = y * GRID_SIZE + (GRID_SIZE - 1)

    A[left, :] = 0
    A[left, left] = 1.0
    b[left] = 1.0

    A[right, :] = 0
    A[right, right] = 1.0
    b[right] = 0.0

A = A.tocsr()
print(f"Matrix built in {time.time() - start_build:.2f} s")

# Solve using Conjugate Gradient
print("Solving transport equation...")
start_solve = time.time()

V, info = splinalg.cg(A, b, rtol=1e-8)
if info != 0:
    print(f"WARNING: Conjugate Gradient did not converge cleanly (info={info}).")

print(f"Solver completed in {time.time() - start_solve:.2f} s")

V_matrix = V.reshape((GRID_SIZE, GRID_SIZE))

# Compute effective resistivity (2D)
dV_right = V_matrix[:, -2] - V_matrix[:, -1]
g_right = g_h[:, -1]

total_current = np.sum(dV_right * g_right)
eff_resistivity_2d = 1.0 / total_current

# ==========================================
# 6. 2D → 3D CORRECTION (BAKKER EMT)
# ==========================================
print("Applying 3D correction (Bakker EMT)...")

sigma_matrix = 1.0 / np.mean(rho_map[sn_mask])
sigma_dispersed = 1.0 / np.mean(rho_map[bi_mask])

cond_ratio = sigma_dispersed / sigma_matrix
c_D = np.mean(bi_mask)

# Empirical shape factor
r = 0.93 + (1.0 / (cond_ratio + 1.03))

eff_conductivity_2d = 1.0 / eff_resistivity_2d
f_2D = eff_conductivity_2d / sigma_matrix

X = 1.0 - ((1.0 - cond_ratio) * c_D)
f_3D = X - ((X - f_2D) / r)

eff_conductivity_3d = f_3D * sigma_matrix
eff_resistivity_3d = 1.0 / eff_conductivity_3d

print("\n====================================")
print(f"2D Resistivity: {eff_resistivity_2d:.2f} µΩ·cm")
print(f"3D Corrected Resistivity: {eff_resistivity_3d:.2f} µΩ·cm")
print("====================================\n")

# ==========================================
# 7. VISUALIZATION
# ==========================================
composition_map = np.zeros((GRID_SIZE, GRID_SIZE))
composition_map[sn_mask] = c_bi_impurity[sn_mask]
composition_map[bi_mask] = c_sn_impurity[bi_mask]

fig, axs = plt.subplots(1, 4, figsize=(32, 8))

axs[0].imshow(phase_matrix, cmap='bone')
axs[0].set_title("Phase Map")
axs[0].axis('off')

im1 = axs[1].imshow(composition_map, cmap='viridis', vmax=0.22)
axs[1].set_title("Impurity Distribution")
axs[1].axis('off')
fig.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(rho_map, cmap='inferno')
axs[2].set_title("Resistivity Map")
axs[2].axis('off')
fig.colorbar(im2, ax=axs[2])

im3 = axs[3].imshow(V_matrix, cmap='magma')
axs[3].set_title(f"Voltage Field (ρ = {eff_resistivity_3d:.1f})")
axs[3].axis('off')
fig.colorbar(im3, ax=axs[3])

plt.tight_layout()
plt.show()