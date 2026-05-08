
import numpy as np
import scipy.sparse as sparse

def build_difference_matrix(N):
    data = np.ones((2, N))
    data[0, :] *= -1
    offsets = [0, 1]
    D = sparse.diags(data, offsets, shape=(N, N))
    D = D.tolil()
    D[-1, 0] = 1  # periodic BC (modify if needed)
    return D.tocsc()

def build_average_matrix(N):
    data = np.ones((2, N)) * 0.5
    offsets = [0, 1]
    A = sparse.diags(data, offsets, shape=(N, N))
    A = A.tolil()
    A[-1, 0] = 0.5
    return A.tocsc()

def build_hodge_operators_pml_drude(
    Nx, Ny,
    dx_array, dy_array,
    dx_dual, dy_dual,
    pml, EPS0, MU0, dt,
    omega_p, gamma
):
    """
    Build Hodge operators for FCI FDTD with split-field PML + Drude model.

    Fields:
    [Hx, Hy, Ezx, Ezy, Jzx, Jzy]
    """

    I_x = sparse.eye(Nx, format='csc')
    I_y = sparse.eye(Ny, format='csc')

    diag_x_inv = sparse.diags(1.0 / dx_array, format='csc')
    diag_y_inv = sparse.diags(1.0 / dy_array, format='csc')

    d_x = build_difference_matrix(Nx)
    d_y = build_difference_matrix(Ny)
    A_x = build_average_matrix(Nx)
    A_y = build_average_matrix(Ny)

    # Hodge operators
    hodge_mu_xx = MU0 * sparse.kron(I_x, A_y)
    hodge_mu_yy = MU0 * sparse.kron(A_x, I_y)
    hodge_eps_zz = EPS0 * sparse.kron(A_x, A_y)

    # Curl operators
    hodge_c_xz = sparse.kron(I_x, diag_y_inv @ d_y)
    hodge_c_yz = sparse.kron(-diag_x_inv @ d_x, I_y)
    hodge_c_zx = sparse.kron(-A_x, diag_y_inv @ d_y)
    hodge_c_zy = sparse.kron(diag_x_inv @ d_x, A_y)

    # PML terms
    S_ex = pml.Sigma_e_x
    S_ey = pml.Sigma_e_y
    S_mx = pml.Sigma_m_x
    S_my = pml.Sigma_m_y

    S_ex_avg = S_ex @ sparse.kron(A_x, A_y)
    S_ey_avg = S_ey @ sparse.kron(A_x, A_y)
    S_mx_avg = S_mx @ sparse.kron(A_x, I_y)
    S_my_avg = S_my @ sparse.kron(I_x, A_y)

    # --- Drude coefficients ---
    a = (1 - gamma * dt / 2) / (1 + gamma * dt / 2)
    b = (EPS0 * omega_p**2 * dt / 2) / (1 + gamma * dt / 2)

    N = Nx * Ny
    I = sparse.eye(N, format='csc')
    A_drude = a * I
    B_drude = b * I

    # --- Build 6x6 block system ---

    # Left matrix
    Mleft = sparse.bmat([
        [hodge_mu_xx/dt + S_my_avg/2, None, -hodge_c_xz/2, -hodge_c_xz/2, None, None],
        [None, hodge_mu_yy/dt + S_mx_avg/2, -hodge_c_yz/2, -hodge_c_yz/2, None, None],
        [None, hodge_c_zy/2, hodge_eps_zz/dt + S_ey_avg/2, None,  I/2, None],
        [hodge_c_zx/2, None, None, hodge_eps_zz/dt + S_ex_avg/2, None, I/2],
        [None, None, -B_drude, None, I, None],
        [None, None, None, -B_drude, None, I]
    ], format='csc')

    # Right matrix
    Mright = sparse.bmat([
        [hodge_mu_xx/dt - S_my_avg/2, None, hodge_c_xz/2, hodge_c_xz/2, None, None],
        [None, hodge_mu_yy/dt - S_mx_avg/2, hodge_c_yz/2, hodge_c_yz/2, None, None],
        [None, -hodge_c_zy/2, hodge_eps_zz/dt - S_ey_avg/2, None, -I/2, None],
        [-hodge_c_zx/2, None, None, hodge_eps_zz/dt - S_ex_avg/2, None, -I/2],
        [None, None, -B_drude, None, A_drude, None],
        [None, None, None, -B_drude, None, A_drude]
    ], format='csc')

    return Mleft, Mright
