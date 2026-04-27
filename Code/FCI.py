import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from datetime import datetime
import os



def create_measurement_folder(base_dir='Code/figures'):
    """Create a new folder with current date and time."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = f"measurement_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")
    return folder_path, timestamp

# ============================================================
# 1. GRID GENERATION FUNCTIONS
# ============================================================

def generate_graded_array(N, dx_min, dx_max, strength, symmetric=True):
    """Create smoothly graded step array: fine at center, coarse at edges."""
    if symmetric:
        x_norm = np.abs(np.linspace(-1, 1, N))
    else:
        x_norm = np.linspace(0, 1, N)
    
    # Shifted tanh: grading starts earlier for more dramatic effect
    profile = 0.5 * (1 + np.tanh(strength * (x_norm - 0.2) / 0.8))
    dx_array = dx_min + (dx_max - dx_min) * profile
    
    if symmetric:
        dx_array = 0.5 * (dx_array + dx_array[::-1])
    
    return dx_array


def create_dual_steps(array):
    """Compute dual steps using harmonic mean."""
    N = len(array)
    dual = np.zeros(N)
    for i in range(N - 1):
        dual[i] = 2.0 * array[i] * array[i+1] / (array[i] + array[i+1])
    dual[-1] = 2.0 * array[-1] * array[0] / (array[-1] + array[0])
    return dual

# ============================================================
# 2. MATRIX CONSTRUCTION FUNCTIONS
# ============================================================

def build_difference_matrix(N):
    """Build periodic first-difference matrix."""
    D = sparse.diags([-1, 1], [0, 1], shape=(N, N), format='lil')
    D[-1, 0] = 1
    return D.tocsc()


def build_average_matrix(N):
    """Build periodic averaging matrix."""
    A = sparse.diags([0.5, 0.5], [0, 1], shape=(N, N), format='lil')
    A[-1, 0] = 0.5
    return A.tocsc()



def build_hodge_operators_pml(Nx, Ny, dx_array, dy_array, dx_dual, dy_dual,
                              pml, EPS0, MU0, dt):
    """
    Build Hodge operators for FCI FDTD with split-field PML.
    
    Fields: [Hx, Hy, Ezx, Ezy] where Ez = Ezx + Ezy
    """
    
    I_x = sparse.eye(Nx, format='csc')
    I_y = sparse.eye(Ny, format='csc')
    N = Nx * Ny
    
    diag_x_inv = sparse.diags(1.0 / dx_array, format='csc')
    diag_y_inv = sparse.diags(1.0 / dy_array, format='csc')
    
    d_x = build_difference_matrix(Nx)
    d_y = build_difference_matrix(Ny)
    A_x = build_average_matrix(Nx)
    A_y = build_average_matrix(Ny)
    
    # Hodge operators (no PML in mu/eps)
    hodge_mu_xx = MU0 * sparse.kron(I_x, A_y)
    hodge_mu_yy = MU0 * sparse.kron(A_x, I_y)
    hodge_eps_zz = EPS0 * sparse.kron(A_x, A_y)
    
    # Curl operators
    hodge_c_xz = sparse.kron(I_x, diag_y_inv @ d_y)
    hodge_c_yz = sparse.kron(-diag_x_inv @ d_x, I_y)
    hodge_c_zx = sparse.kron(-A_x, diag_y_inv @ d_y)
    hodge_c_zy = sparse.kron(diag_x_inv @ d_x, A_y)

    S_ex_averaged = pml.Sigma_e_x @ sparse.kron(A_x, A_y)
    S_my_averaged = pml.Sigma_m_y @ sparse.kron(I_x, A_y)
    S_ey_averaged = pml.Sigma_e_x @ sparse.kron(A_x, A_y)
    S_mx_averaged = pml.Sigma_m_y @ sparse.kron(A_x, I_y)
    
    # --- Build 4x4 block system for [Hx, Hy, Ezx, Ezy] ---
    
    # Left matrix (implicit)
    Mleft = sparse.bmat([
        [hodge_mu_xx/dt + S_my_averaged/2, None,                     -hodge_c_xz/2,            -hodge_c_xz/2           ],
        [None,                     hodge_mu_yy/dt +S_mx_averaged/2,  -hodge_c_yz/2,            -hodge_c_yz/2           ],
        [None,             hodge_c_zy/2,                     hodge_eps_zz/dt + S_ex_averaged/2, None                    ],
        [hodge_c_zx/2,                     None,             None,                     hodge_eps_zz/dt + S_ey_averaged/2]
    ], format='csc')
    
    # Right matrix (explicit)
    Mright = sparse.bmat([
        [hodge_mu_xx/dt -S_my_averaged/2, None,                     hodge_c_xz/2,             hodge_c_xz/2            ],
        [None,                     hodge_mu_yy/dt - S_mx_averaged/2,  hodge_c_yz/2,             hodge_c_yz/2            ],
        [None,            -hodge_c_zy/2,                     hodge_eps_zz/dt - S_ex_averaged/2, None                    ],
        [-hodge_c_zx/2,                    None,            None,                     hodge_eps_zz/dt - S_ey_averaged/2]
    ], format='csc')
    
    return Mleft, Mright

def get_source_value(t, dt):
    """Gaussian pulse source."""
    t0 = 20 * dt
    sigma = 5 * dt
    return   np.exp(-((t - t0)**2) / (2 * sigma**2))




class FCIFDTD:
    """Fully Collocated Implicit FDTD solver."""
    
    def __init__(self, Nx=201, Ny=201, Nt=200, 
                 dx_min=None, dx_max_factor=5.0, grading_strength=3.0,
                 EPS0= 8.854e-12, MU0= 4 * np.pi * 1e-7, CFL=1.0):
        
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.N = Nx * Ny
        self.EPS0 = EPS0
        self.MU0 = MU0
        self.C0 = 1.0 / np.sqrt(EPS0 * MU0)
        
        # Source parameters
        sigma = 0.03 * 1e-1
        self.f_max = (5 / sigma) / (2 * np.pi)
        lambda_min = self.C0 / self.f_max
        
        # Grid spacing
        if dx_min is None:
            dx_min = lambda_min / 50
        dx_max = dx_max_factor * dx_min
        
        self.dx_array = generate_graded_array(Nx, dx_min, dx_max, grading_strength)
        self.dy_array = generate_graded_array(Ny, dx_min, dx_max, grading_strength)
        
        self.dx_dual = create_dual_steps(self.dx_array)
        self.dy_dual = create_dual_steps(self.dy_array)
        
        # Time step (based on minimum spacing for stability)
        self.dt = CFL / (self.C0 * np.sqrt(1/dx_min**2 + 1/dx_min**2))
        
        # Source location
        self.source_x = Nx // 2
        self.source_y = Ny // 2
        self.source_index = self.source_x * Ny + self.source_y
        
        # Build system matrices
        print("Building system matrices...")
        self.Mleft, self.Mright = build_hodge_operators(
            Nx, Ny, self.dx_array, self.dy_array, 
            self.dx_dual, self.dy_dual, EPS0, MU0, self.dt
        )
        
        print("Factorizing left matrix...")
        self.solver = sparse.linalg.factorized(self.Mleft)
        
        # Initialize fields
        self.fields = np.zeros(3 * self.N)
        self.Ez_history = []
        
        print(f"Grid: {Nx}x{Ny}, dt={self.dt:.6e}")
        print(f"dx_min={dx_min:.6e}, dx_max={dx_max:.6e}")
    
    def step(self, n):
        """Advance one time step."""
        fields_old = self.fields.copy()
        rhs = self.Mright @ fields_old
        rhs[2*self.N + self.source_index] -= get_source_value(n * self.dt, self.dt)
        self.fields = self.solver(rhs)
        return self.fields
    
    def run(self, save_every=2):
        """Run full simulation."""
        print(f"Running {self.Nt} steps...")
        for n in range(self.Nt):
            if n % 10 == 0:
                print(f"  Step {n}/{self.Nt}")
            self.step(n)
            if n % save_every == 0:
                Ez = self.fields[2*self.N:].reshape((self.Nx, self.Ny))
                self.Ez_history.append(Ez.copy())
        print("Done!")
        return self.Ez_history
    
    def get_Ez(self):
        """Get current Ez field."""
        return self.fields[2*self.N:].reshape((self.Nx, self.Ny))

class FCIFDTDPML:
    """
    Fully Collocated Implicit FDTD with split-field PML.
    Fields: [Hx, Hy, Ezx, Ezy] where Ez = Ezx + Ezy
    """
    
    def __init__(self, Nx=201, Ny=201, Nt=200,
                 dx_min=None, dx_max_factor=5.0, grading_strength=3.0,
                 pml_width_x=0.0, pml_width_y=0.0,
                 EPS0=8.854e-12, MU0=4*np.pi*1e-7, CFL=1.0):
        
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.N = Nx * Ny
        self.EPS0 = EPS0
        self.MU0 = MU0
        self.C0 = 1.0 / np.sqrt(EPS0 * MU0)
        
        # Source parameters
        sigma = 0.03 * 1e-1
        self.f_max = (5 / sigma) / (2 * np.pi)
        lambda_min = self.C0 / self.f_max
        
        # Grid spacing
        if dx_min is None:
            dx_min = lambda_min / 50
        dx_max = dx_max_factor * dx_min
        
        self.dx_array = generate_graded_array(Nx, dx_min, dx_max, grading_strength)
        self.dy_array = generate_graded_array(Ny, dx_min, dx_max, grading_strength)
        
        self.dx_dual = create_dual_steps(self.dx_array)
        self.dy_dual = create_dual_steps(self.dy_array)
        
        # Time step
        self.dt = CFL / (self.C0 * np.sqrt(1/dx_min**2 + 1/dx_min**2))
        
        # Source location
        self.source_x = Nx // 2
        self.source_y = Ny // 2
        self.source_index = self.source_x * Ny + self.source_y
        
        # PML
        self.pml = PMLProfile(Nx, Ny, self.dx_array, self.dy_array,
                              pml_width_x, pml_width_y, EPS0, MU0)
        
        
        print("Building PML system matrices...")
        self.Mleft, self.Mright = build_hodge_operators_pml(
            Nx, Ny, self.dx_array, self.dy_array,
            self.dx_dual, self.dy_dual, self.pml,
            EPS0, MU0, self.dt
        )
        
        print("Factorizing...")
        self.solver = sparse.linalg.factorized(self.Mleft)
        
        # Initialize 4 fields: [Hx, Hy, Ezx, Ezy]
        self.fields = np.zeros(4 * self.N)
        self.Ez_history = []
        
        print(f"Grid: {Nx}x{Ny}, dt={self.dt:.6e}")
        print(f"PML: wx={pml_width_x}, wy={pml_width_y}")
    
    def step(self, n):
        fields_old = self.fields.copy()
        rhs = self.Mright @ fields_old
        

        source = get_source_value(n * self.dt, self.dt)
        rhs[2*self.N + self.source_index] += source / 2
        rhs[3*self.N + self.source_index] -= source / 2
        
        self.fields = self.solver(rhs)
        return self.fields
    
    def run(self, save_every=2):
        print(f"Running {self.Nt} steps...")
        for n in range(self.Nt):
            if n % 10 == 0:
                print(f"  Step {n}/{self.Nt}")
            self.step(n)
            if n % save_every == 0:
                # Reconstruct Ez = Ezx + Ezy
                Ezx = self.fields[2*self.N:3*self.N].reshape((self.Nx, self.Ny))
                Ezy = self.fields[3*self.N:].reshape((self.Nx, self.Ny))
                Ez = Ezx + Ezy
                self.Ez_history.append(Ez.copy())
        print("Done!")
        return self.Ez_history
    
    def get_Ez(self):
        Ezx = self.fields[2*self.N:3*self.N].reshape((self.Nx, self.Ny))
        Ezy = self.fields[3*self.N:].reshape((self.Nx, self.Ny))
        return Ezx + Ezy

class PMLProfile:
    """Creates PML layers.
    
    
    """  
    #todo create for x and y different animations
    def __init__(self, Nx, Ny, dx_array, dy_array, 
                 pml_width_x, pml_width_y,
                 EPS0=8.854e-12, MU0=4*np.pi*1e-7, m=3.5):
        
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny
        
        
        self.Nx_pml = int(pml_width_x / np.mean(dx_array)) if pml_width_x > 0 else 0
        self.Ny_pml = int(pml_width_y / np.mean(dy_array)) if pml_width_y > 0 else 0
        
        eta0 = np.sqrt(MU0 / EPS0)

        sigma_e_x = np.zeros(Nx)
        sigma_m_x = np.zeros(Nx)
        
        if self.Nx_pml > 0:
            sigma_max_x = 10*(m + 1) / (2 * eta0 * pml_width_x)
            
            # Indices for the left and right PML regions
            i_left = np.arange(self.Nx_pml)
            i_right = np.arange(Nx - self.Nx_pml, Nx)
            
            # Left edge profile
            factor_l = ((self.Nx_pml - i_left) / self.Nx_pml) ** m
            sigma_e_x[:self.Nx_pml] = sigma_max_x * factor_l
            sigma_m_x[:self.Nx_pml] = sigma_max_x * (MU0 / EPS0) * factor_l
            
            # Right edge profile
            factor_r = ((i_right - (Nx - self.Nx_pml) + 1) / self.Nx_pml) ** m
            sigma_e_x[Nx - self.Nx_pml:] = sigma_max_x * factor_r
            sigma_m_x[Nx - self.Nx_pml:] = sigma_max_x * (MU0 / EPS0) * factor_r

       
        sigma_e_y = np.zeros(Ny)
        sigma_m_y = np.zeros(Ny)

        if self.Ny_pml > 0:
            sigma_max_y = 10*(m + 1) / (2 * eta0 * pml_width_y)
            
            j_bottom = np.arange(self.Ny_pml)
            j_top = np.arange(Ny - self.Ny_pml, Ny)
            
            # Bottom edge profile
            factor_b = ((self.Ny_pml - j_bottom) / self.Ny_pml) ** m
            sigma_e_y[:self.Ny_pml] = sigma_max_y * factor_b
            sigma_m_y[:self.Ny_pml] = sigma_max_y * (MU0 / EPS0) * factor_b
            
            # Top edge profile
            factor_t = ((j_top - (Ny - self.Ny_pml) + 1) / self.Ny_pml) ** m
            sigma_e_y[Ny - self.Ny_pml:] = sigma_max_y * factor_t
            sigma_m_y[Ny - self.Ny_pml:] = sigma_max_y * (MU0 / EPS0) * factor_t
        
       
        self.sigma_e_x_2d = np.tile(sigma_e_x[:, np.newaxis], (1, Ny))
        self.sigma_e_y_2d = np.tile(sigma_e_y[np.newaxis, :], (Nx, 1))
        self.sigma_m_x_2d = np.tile(sigma_m_x[:, np.newaxis], (1, Ny))
        self.sigma_m_y_2d = np.tile(sigma_m_y[np.newaxis, :], (Nx, 1))
        
        
        self.sigma_e_x_flat = self.sigma_e_x_2d.ravel()
        self.sigma_e_y_flat = self.sigma_e_y_2d.ravel()
        self.sigma_m_x_flat = self.sigma_m_x_2d.ravel()
        self.sigma_m_y_flat = self.sigma_m_y_2d.ravel()
        
        
        self.Sigma_e_x = sparse.diags(self.sigma_e_x_flat, format='csc')
        self.Sigma_e_y = sparse.diags(self.sigma_e_y_flat, format='csc')
        self.Sigma_m_x = sparse.diags(self.sigma_m_x_flat, format='csc')
        self.Sigma_m_y = sparse.diags(self.sigma_m_y_flat, format='csc')
        
        print(f"PML: Nx_pml={self.Nx_pml}, Ny_pml={self.Ny_pml}")
        print(f"Max sigma_e_x={self.sigma_e_x_flat.max():.4e}")
        
    def plot_profiles(self, folder_path, timestamp):
        """Visualize PML conductivity profiles."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        im1 = axes[0,0].imshow(self.sigma_e_x_2d, origin='lower', cmap='hot')
        axes[0,0].set_title('sigma_e_x (electric, x-PML)')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(self.sigma_e_y_2d, origin='lower', cmap='hot')
        axes[0,1].set_title('sigma_e_y (electric, y-PML)')
        plt.colorbar(im2, ax=axes[0,1])
        
        im3 = axes[1,0].imshow(self.sigma_m_x_2d, origin='lower', cmap='hot')
        axes[1,0].set_title('sigma_m_x (magnetic, x-PML)')
        plt.colorbar(im3, ax=axes[1,0])
        
        im4 = axes[1,1].imshow(self.sigma_m_y_2d, origin='lower', cmap='hot')
        axes[1,1].set_title('sigma_m_y (magnetic, y-PML)')
        plt.colorbar(im4, ax=axes[1,1])
        
        plt.tight_layout()
        filepath = os.path.join(folder_path, f'pml_profiles_{timestamp}.png')
        plt.savefig(filepath, dpi=150)
        plt.show()
        print(f"Saved PML profiles: {filepath}")
        return filepath


def plot_grid(dx_array, dy_array, Nx, Ny, folder_path, timestamp):
    """
    Visualize non-uniform grid with clear fine vs coarse comparison.
    FIXED: Shows true edge coarseness correctly.
    """
    x_pos = np.concatenate([[0], np.cumsum(dx_array)])
    y_pos = np.concatenate([[0], np.cumsum(dy_array)])
    center_x = x_pos[Nx//2]
    center_y = y_pos[Ny//2]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax1 = axes[0, 0]
    step = max(1, Nx//40)
    for i in range(0, Nx+1, step):
        ax1.axvline(x=x_pos[i], color='blue', linewidth=0.4, alpha=0.6)
    for j in range(0, Ny+1, step):
        ax1.axhline(y=y_pos[j], color='red', linewidth=0.4, alpha=0.6)
    ax1.plot(center_x, center_y, 'go', markersize=10, label='Source')
    ax1.set_aspect('equal')
    ax1.set_title('Full Grid Overview')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.legend()
    

    ax2 = axes[0, 1]
    z = 5
    i_start, i_end = Nx//2 - z, Nx//2 + z + 1
    j_start, j_end = Ny//2 - z, Ny//2 + z + 1
    
    for i in range(i_start, i_end):
        ax2.axvline(x=x_pos[i], color='blue', linewidth=1.0)
    for j in range(j_start, j_end):
        ax2.axhline(y=y_pos[j], color='red', linewidth=1.0)
    
    # Draw cell boundaries
    for i in range(i_start, i_end-1):
        for j in range(j_start, j_end-1):
            rect = plt.Rectangle((x_pos[i], y_pos[j]), 
                           x_pos[i+1]-x_pos[i], y_pos[j+1]-y_pos[j],
                           fill=False, edgecolor='gray', linewidth=0.5)
            ax2.add_patch(rect)
    
    # Label a center cell with its size
    cx, cy = Nx//2, Ny//2
    ax2.text(x_pos[cx] + dx_array[cx]/2, y_pos[cy] + dy_array[cy]/2,
             f'{dx_array[cx]:.4f}', ha='center', va='center', fontsize=8, color='green')
    
    ax2.plot(center_x, center_y, 'go', markersize=10)
    ax2.set_xlim([x_pos[i_start], x_pos[i_end]])
    ax2.set_ylim([y_pos[j_start], y_pos[j_end]])
    ax2.set_aspect('equal')
    ax2.set_title(f'CENTER: Fine Grid\nCell size ≈ {dx_array[Nx//2]:.4f}')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    
    ax3 = axes[0, 2]
    
  
    edge_z = z  # Same cell count as center
    i_start_e, i_end_e = 0, 2*edge_z + 1  # 0 to 10 (11 lines, 10 cells)
    j_start_e, j_end_e = 0, 2*edge_z + 1
    
    for i in range(i_start_e, i_end_e):
        ax3.axvline(x=x_pos[i], color='blue', linewidth=1.0)
    for j in range(j_start_e, j_end_e):
        ax3.axhline(y=y_pos[j], color='red', linewidth=1.0)
    
    # Draw cell boundaries and label
    for i in range(i_start_e, i_end_e-1):
        for j in range(j_start_e, j_end_e-1):
            rect = plt.Rectangle((x_pos[i], y_pos[j]), 
                           x_pos[i+1]-x_pos[i], y_pos[j+1]-y_pos[j],
                           fill=False, edgecolor='gray', linewidth=0.5)
            ax3.add_patch(rect)
    
    # Label first cell with its size
    ax3.text(x_pos[0] + dx_array[0]/2, y_pos[0] + dy_array[0]/2,
             f'{dx_array[0]:.4f}', ha='center', va='center', fontsize=8, color='red')
    
    # Calculate physical width comparison
    center_phys_width = x_pos[Nx//2 + z] - x_pos[Nx//2 - z]
    edge_phys_width = x_pos[2*edge_z] - x_pos[0]
    
    ax3.set_xlim([x_pos[0], x_pos[2*edge_z]])
    ax3.set_ylim([y_pos[0], y_pos[2*edge_z]])
    ax3.set_aspect('equal')
    ax3.set_title(f'EDGE: Coarse Grid\nCell size ≈ {dx_array[0]:.4f}\n'
                  f'Same {2*edge_z} cells = {edge_phys_width/center_phys_width:.1f}x wider')
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    
    # --- Row 2: Quantitative analysis ---
    
    # Step size vs index
    ax4 = axes[1, 0]
    ax4.plot(dx_array, 'b-', linewidth=1.5, label='dx')
    ax4.plot(dy_array, 'r--', linewidth=1.5, label='dy')
    ax4.axvline(x=Nx//2, color='green', linestyle=':', alpha=0.7, label='Center')
    ax4.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='Edge')
    ax4.set_xlabel('Cell Index')
    ax4.set_ylabel('Step Size')
    ax4.set_title('Step Size vs Index')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Step size vs position (log scale)
    ax5 = axes[1, 1]
    x_mid = (x_pos[:-1] + x_pos[1:]) / 2
    ax5.semilogy(x_mid, dx_array, 'b-', linewidth=1.5)
    ax5.axhline(y=dx_array.min(), color='green', linestyle='--', alpha=0.7, label=f'dx_min={dx_array.min():.4f}')
    ax5.axhline(y=dx_array.max(), color='red', linestyle='--', alpha=0.7, label=f'dx_max={dx_array.max():.4f}')
    ax5.set_xlabel('Position')
    ax5.set_ylabel('Step Size (log)')
    ax5.set_title('Grading Profile (Log Scale)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Histogram with clear min/max markers
    ax6 = axes[1, 2]
    ax6.hist(dx_array, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax6.axvline(x=dx_array.min(), color='green', linestyle='--', linewidth=2, label=f'Min={dx_array.min():.4f}')
    ax6.axvline(x=dx_array.max(), color='red', linestyle='--', linewidth=2, label=f'Max={dx_array.max():.4f}')
    ax6.axvline(x=np.mean(dx_array), color='orange', linestyle='-', linewidth=2, label=f'Mean={np.mean(dx_array):.4f}')
    ax6.set_xlabel('Step Size')
    ax6.set_ylabel('Number of Cells')
    ax6.set_title('Distribution of Step Sizes')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = os.path.join(folder_path, f'grid_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filepath}")
    
    # Print stats
    print(f"\n{'='*50}")
    print("GRID STATISTICS")
    print(f"{'='*50}")
    print(f"  Nx = {Nx}, Ny = {Ny}")
    print(f"  dx_min = {dx_array.min():.6f}  (center)")
    print(f"  dx_max = {dx_array.max():.6f}  (edge)")
    print(f"  Ratio = {dx_array.max()/dx_array.min():.2f}")
    print(f"  Center {2*z} cells width = {center_phys_width:.4f}")
    print(f"  Edge {2*z} cells width = {edge_phys_width:.4f}")
    print(f"  Edge/Center width ratio = {edge_phys_width/center_phys_width:.2f}")
    print(f"  Total domain = {x_pos[-1]:.4f} x {y_pos[-1]:.4f}")
    print(f"{'='*50}")
    
    return filepath


def create_animation(Ez_history, folder_path, timestamp, fps=10):
    """Create GIF animation and save to measurement folder."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    vmax = max(np.max(np.abs(E)) for E in Ez_history) * 0.5
    im = ax.imshow(Ez_history[0], cmap='RdBu', origin='lower',
                   vmin=-vmax, vmax=vmax)
    ax.set_title('Step 0')
    plt.colorbar(im, ax=ax, label='Ez')
    
    filepath = os.path.join(folder_path, f'simulation_{timestamp}.gif')
    writer = PillowWriter(fps=fps)
    
    with writer.saving(fig, filepath, dpi=100):
        for i, Ez in enumerate(Ez_history):
            im.set_array(Ez)
            ax.set_title(f'Step {i*2}')
            writer.grab_frame()
            if i % 20 == 0:
                print(f"  Writing frame {i}/{len(Ez_history)}")
    
    plt.close()
    print(f"Saved animation: {filepath}")
    return filepath





def save_metadata(sim, folder_path, timestamp, Nt, save_every,grading_strength):
    """Save simulation parameters to text file."""
    metadata = {
        'timestamp': timestamp,
        'Nx': sim.Nx,
        'Ny': sim.Ny,
        'Nt': Nt,
        'dt': sim.dt,
        'dx_min': sim.dx_array.min(),
        'dx_max': sim.dx_array.max(),
        'dx_ratio': sim.dx_array.max() / sim.dx_array.min(),
        'source_position': (sim.source_x, sim.source_y),
        'save_every': save_every,
        'total_frames': len(sim.Ez_history),
        'grading_strenght': grading_strength
    }
    
    filepath = os.path.join(folder_path, f'metadata_{timestamp}.txt')
    with open(filepath, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Saved metadata: {filepath}")
    return filepath



if __name__ == "__main__":
    #geen zorgen maken hier runt je programma
    folder_path, timestamp = create_measurement_folder(base_dir='measurementsFCI')
    
    Nx, Ny = 101, 101
    
    sigma = 0.03 * 1e-1
    f_max = (5 / sigma) / (2 * np.pi)
    C0 = 3*1e8
    lambda_min = C0 / f_max
    dx_min = lambda_min / 50
    dx_max = 5 * dx_min
    grading_strength = 1
    
    dx_array = generate_graded_array(Nx, dx_min, dx_max, grading_strength)
    dy_array = generate_graded_array(Ny, dx_min, dx_max, grading_strength)
    
    simulation_width = np.sum(dx_array)
    simulation_height = np.sum(dy_array)

    plot_grid(dx_array, dy_array, Nx, Ny, folder_path, timestamp)
    
   
    print("\n" + "="*50)
    print("STARTING FCI FDTD WITH PML")
    print("="*50)
    

    pml_wx = simulation_width * 0.2
    pml_wy = simulation_height * 0.2
    
    sim = FCIFDTDPML(Nx=Nx, Ny=Ny, Nt=500,
                     dx_min=dx_min, dx_max_factor=5.0,
                     grading_strength=grading_strength,
                     pml_width_x=pml_wx, pml_width_y=pml_wy,
                    ) 
    sim.pml.plot_profiles(folder_path, timestamp)
    
    Ez_history = sim.run(save_every=2)
    
    create_animation(Ez_history, folder_path, timestamp, fps=10)
    save_metadata(sim, folder_path, timestamp, Nt=300, save_every=2, 
                 grading_strength=grading_strength)

    print(f"\nALL FILES SAVED TO: {folder_path}")
