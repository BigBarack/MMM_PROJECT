import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse.linalg import LinearOperator, cg
Nx, Ny = 101, 101

EPS0, MU0 = 1,1
#EPS0, MU0 = 8.854e-12, 4 * np.pi * 1e-7
C0 = 1 / np.sqrt(EPS0 * MU0)

sigma = 0.03*1e-1
f_max = (5/sigma)/(2*np.pi)

lambda_min = C0 / f_max       # minimum wavelength
dx = lambda_min / 50         # spatial step
dy = dx
CFL = 1
dt = CFL / (C0*np.sqrt(1/dx**2 + 1/dy**2)) 
N = Nx * Ny


Nt = 400


# Center of the grid
source_x, source_y = Nx // 2, Ny // 2
source_index = source_x * Ny + source_y  # Flattened index for (source_x, source_y)

def get_source_value(t, dt):
    # Gaussian pulse
    t0 = 20 * dt  # Peak time
    sigma = 5 * dt # Width
    return 1e6 *np.exp(-((t - t0)**2) / (2 * sigma**2))


def plot_matrix(matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('d_x Matrix Visualization')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

Ez = np.zeros((Nx,Ny))
Hx = Ez.copy()
Hy = Ez.copy()

Ezflat = Ez.ravel()
Hxflat = Hx.ravel()
Hyflat = Hy.ravel()

fields = np.concatenate([Hxflat, Hyflat, Ezflat])


d_x_circulant = sparse.diags([-1, 1], [0, 1], shape=(Nx, Nx), format='lil')
d_x_circulant[-1, 0] = 1  # Wrap-around: last row connects to first column


d_y_circulant = sparse.diags([-1, 1], [0, 1], shape=(Ny, Ny), format='lil')
d_y_circulant[-1, 0] = 1


Ax_circulant = sparse.diags([0.5, 0.5], [0, 1], shape=(Nx, Nx), format='lil')
Ax_circulant[-1, 0] = 0.5  # Wrap-around


Ay_circulant = sparse.diags([0.5, 0.5], [0, 1], shape=(Ny, Ny), format='lil')
Ay_circulant[-1, 0] = 0.5


I_x = sparse.eye(Nx,format='csc')
I_y = sparse.eye(Ny,format='csc')

diag_step_x = sparse.diags([dx],shape=(Nx,Nx),format='csc')
diag_step_y = sparse.diags([dy],shape=(Ny,Ny),format='csc')

dx_array = dx* np.ones((Nx,))

diag_step_x = sparse.diags(dx_array, format='csc')
diag_step_y = sparse.diags(dx_array,format='csc')

diag_step_x_inverse = sparse.diags(1.0 / dx_array, format='csc')
diag_step_y_inverse = sparse.diags(1.0 / dx_array, format='csc')
diag_step_xx = sparse.kron(diag_step_x,I_y)
diag_step_xx_inverse = sparse.kron(diag_step_x_inverse,I_y)
diag_step_yy = sparse.kron(I_x,diag_step_y)
diag_step_yy_inverse = sparse.kron(I_x,diag_step_y_inverse)

diag_step_x_dual = sparse.diags(dx_array, format='csc')
diag_step_y_dual = sparse.diags(dx_array,format='csc')

diag_step_x_inverse_dual = sparse.diags(1.0 / dx_array, format='csc')
diag_step_y_inverse_dual = sparse.diags(1.0 / dx_array, format='csc')
diag_step_xx_dual = sparse.kron(diag_step_x,I_y)
diag_step_xx_inverse_dual = sparse.kron(diag_step_x_inverse,I_y)
diag_step_yy_dual = sparse.kron(I_x,diag_step_y)
diag_step_yy_inverse_dual = sparse.kron(I_x,diag_step_y_inverse)

hodge_mu_xx = MU0 * sparse.kron(I_x, Ay_circulant)
hodge_mu_yy = MU0 * sparse.kron(Ax_circulant, I_y)
hodge_eps_zz = EPS0 * sparse.kron(Ax_circulant, Ay_circulant)

hodge_c_xz = sparse.kron(I_x, diag_step_y_inverse @ d_y_circulant)
hodge_c_yz = sparse.kron(-diag_step_x_inverse @ d_x_circulant, I_y)

hodge_c_zx = sparse.kron(-Ax_circulant, diag_step_y_inverse @ d_y_circulant)
hodge_c_zy = sparse.kron(diag_step_x_inverse @ d_x_circulant, Ay_circulant)


Mleft = sparse.bmat([[hodge_mu_xx/dt,None,-hodge_c_xz/2],
                     [None,hodge_mu_yy/dt,-hodge_c_yz/2],
                     [hodge_c_zx/2,hodge_c_zy/2,hodge_eps_zz/dt]])

Mright = sparse.bmat([[hodge_mu_xx/dt,None,hodge_c_xz/2],
                     [None,hodge_mu_yy/dt,hodge_c_yz/2],
                     [-hodge_c_zx/2,-hodge_c_zy/2,hodge_eps_zz/dt]])


Mleft_solver =sparse.linalg.factorized(Mleft)

def update(fields,current_time):
    Hx = fields[:N].copy()
    Hy = fields[N:2*N].copy()
    Ez = fields[2*N:].copy()

    
   

    
    fields_old = np.concatenate([Hx, Hy, Ez])
    rhs = Mright @ fields_old

    # inject source into RHS (Ez block)
    rhs[2*N + source_index] += get_source_value(current_time, dt)

    # Solve for next step
    fields_new = Mleft_solver(rhs)

    return fields_new

fig , ax = plt.subplots(figsize=(6,6))

frames= []
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(Ezflat.reshape((Nx, Ny)), cmap='RdBu', origin='lower', vmin=-1, vmax=1)

writer = animation.FFMpegWriter(fps=20)

with writer.saving(fig, "simulation3.mp4", dpi=100):
    for n in range(Nt):
        print(f"frame{n}/Nt")
        fields = update(fields, n * dt)

        if n % 2 == 0:
            Ez= fields[2*N:].reshape((Nx, Ny))
            im.set_array(Ez)
            ax.set_title(f"Step {n}")
            writer.grab_frame()
    print("done")








