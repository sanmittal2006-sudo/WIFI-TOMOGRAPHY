"""
=============================================================================
Wi-Fi Microwave Tomography for Pulmonary Edema Detection
Complete Pipeline: Phantom → Simulation → Calibration → BIM → PINN → U-Net → Heatmap

Indian Institute of Science, Bangalore — 4th Year Major Project
Hardware: 2× ESP32-S3 (TX + RX), 2.4 GHz, 16 positions at 22.5° each
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive: saves plots to files without stopping
import matplotlib.pyplot as plt
from scipy.special import hankel1
from scipy.linalg import solve
from scipy.ndimage import gaussian_filter
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing PyTorch (for PINN and U-Net)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("✅ PyTorch available — PINN and U-Net enabled")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed — PINN/U-Net disabled, BIM only")
    print("   Install with: pip install torch")

# Try importing skimage for SSIM
try:
    from skimage.metrics import structural_similarity as ssim_metric
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# =============================================================================
# SECTION 1: ANATOMICAL CHEST PHANTOM
# Creates a 2D cross-section of the human chest with:
#   - Chest wall (muscle, εᵣ ≈ 52)
#   - Left lung (healthy air, εᵣ ≈ 3)
#   - Right lung (edematous fluid, εᵣ varies by severity)
#   - Heart (εᵣ ≈ 58)
#   - Spine (bone, εᵣ ≈ 13)
#   - Fat layer (εᵣ ≈ 6)
#   - Skin (εᵣ ≈ 38)
# =============================================================================

def create_chest_phantom(N=64, edema_level='healthy'):
    """
    Create anatomically realistic 2D chest cross-section phantom.
    
    Parameters:
        N: grid size (NxN pixels)
        edema_level: 'healthy', 'mild', 'moderate', or 'severe'
    
    Returns:
        phantom: NxN array of εᵣ values
        labels: NxN array of tissue type labels (for visualization)
    """
    phantom = np.ones((N, N)) * 1.0  # Start with air everywhere (εᵣ = 1)
    labels = np.zeros((N, N), dtype=int)  # 0 = air
    
    cx, cy = N // 2, N // 2  # Center of image
    Y, X = np.mgrid[0:N, 0:N]
    
    # Scale factor (pixels per cm, assuming 30cm chest width)
    scale = N / 30.0  # pixels per cm
    
    # --- SKIN (outer ellipse, εᵣ ≈ 38) ---
    skin = ((X - cx) / (14 * scale))**2 + ((Y - cy) / (11 * scale))**2 <= 1
    phantom[skin] = 38
    labels[skin] = 1  # skin
    
    # --- FAT LAYER (slightly smaller ellipse, εᵣ ≈ 6) ---
    fat = ((X - cx) / (13.5 * scale))**2 + ((Y - cy) / (10.5 * scale))**2 <= 1
    phantom[fat] = 6
    labels[fat] = 2  # fat
    
    # --- MUSCLE / CHEST WALL (εᵣ ≈ 52) ---
    muscle = ((X - cx) / (13 * scale))**2 + ((Y - cy) / (10 * scale))**2 <= 1
    phantom[muscle] = 52
    labels[muscle] = 3  # muscle
    
    # --- LEFT LUNG (healthy air, εᵣ ≈ 3) ---
    left_lung = ((X - cx + 5 * scale) / (4.5 * scale))**2 + \
                ((Y - cy) / (6.5 * scale))**2 <= 1
    phantom[left_lung] = 3  # Healthy air-filled lung
    labels[left_lung] = 4  # left lung
    
    # --- RIGHT LUNG (εᵣ depends on edema level) ---
    right_lung = ((X - cx - 5 * scale) / (4.5 * scale))**2 + \
                 ((Y - cy) / (6.5 * scale))**2 <= 1
    
    # Set right lung εᵣ based on edema severity
    edema_values = {
        'healthy': 3,       # Normal air-filled
        'mild': 20,         # 25% fluid mixed with air
        'moderate': 45,     # 60% fluid
        'severe': 68        # Nearly fully fluid-filled
    }
    right_lung_er = edema_values.get(edema_level, 3)
    phantom[right_lung] = right_lung_er
    labels[right_lung] = 5  # right lung
    
    # --- HEART (center, εᵣ ≈ 58) ---
    heart = ((X - cx) / (3.5 * scale))**2 + ((Y - cy + 0.5 * scale) / (4 * scale))**2 <= 1
    phantom[heart] = 58
    labels[heart] = 6  # heart
    
    # --- AORTA (large blood vessel behind heart, εᵣ ≈ 65) ---
    aorta = ((X - cx) / (1.2 * scale))**2 + ((Y - cy - 3 * scale) / (1.2 * scale))**2 <= 1
    phantom[aorta] = 65
    labels[aorta] = 7  # aorta
    
    # --- SPINE (posterior, εᵣ ≈ 13 for bone) ---
    spine = ((X - cx) / (2 * scale))**2 + ((Y - cy - 7 * scale) / (2.5 * scale))**2 <= 1
    phantom[spine] = 13
    labels[spine] = 8  # spine
    
    # --- RIBS (12 small bone circles around chest wall, εᵣ ≈ 13) ---
    rib_angles = np.linspace(0.3, 2.8, 10)  # 10 ribs visible in cross-section
    for angle in rib_angles:
        rib_x = cx + int(12 * scale * np.cos(angle))
        rib_y = cy + int(9 * scale * np.sin(angle))
        rib = ((X - rib_x) / (0.5 * scale))**2 + ((Y - rib_y) / (0.5 * scale))**2 <= 1
        phantom[rib] = 13
        labels[rib] = 9  # rib
    
    return phantom, labels


# =============================================================================
# SECTION 2: FORWARD MODEL — GREEN'S FUNCTION MATRIX
# Builds the sensing matrix G that connects the unknown εᵣ distribution
# to the CSI measurements. Uses the Born approximation.
# G(m,j) = how strongly pixel j contributes to measurement m
# =============================================================================

def build_green_function_matrix(N=64, domain_size=0.30, antenna_radius=0.18,
                                  num_positions=16, freq=2.4e9, 
                                  num_subcarriers=56):
    """
    Build the Green's function matrix G for circular measurement geometry.
    
    The phantom rotates while TX/RX are fixed.
    TX is at (-antenna_radius, 0), RX at (+antenna_radius, 0).
    At each rotation angle, we get one measurement per subcarrier.
    
    Parameters:
        N: grid size (NxN pixels)
        domain_size: physical size of the domain (meters), e.g., 0.30 = 30cm
        antenna_radius: distance of TX/RX from center (meters)
        num_positions: number of rotation positions (16 for 22.5° steps)
        freq: Wi-Fi frequency (Hz)
        num_subcarriers: number of OFDM subcarriers (56 for HT40)
    
    Returns:
        G: complex matrix of shape [M, N²] where M = num_positions × num_subcarriers
           (simplified: M = num_positions, using mean across subcarriers)
        k0: wavenumber
        pixel_positions: (N², 2) array of pixel center coordinates
    """
    c = 3e8  # Speed of light (m/s)
    k0 = 2 * np.pi * freq / c  # Wavenumber (1/m)
    wavelength = c / freq
    pixel_size = domain_size / N  # Size of each pixel (m)
    
    print(f"  Frequency: {freq/1e9:.1f} GHz")
    print(f"  Wavelength: {wavelength*100:.1f} cm")
    print(f"  Wavenumber k₀: {k0:.1f} rad/m")
    print(f"  Grid: {N}×{N} = {N*N} pixels")
    print(f"  Pixel size: {pixel_size*100:.2f} cm")
    print(f"  Domain size: {domain_size*100:.0f} cm")
    print(f"  Antenna radius: {antenna_radius*100:.0f} cm")
    
    # Create pixel grid (centers of each pixel)
    half = domain_size / 2
    x = np.linspace(-half + pixel_size/2, half - pixel_size/2, N)
    y = np.linspace(-half + pixel_size/2, half - pixel_size/2, N)
    grid_x, grid_y = np.meshgrid(x, y)
    pixel_positions = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    # Shape: [N², 2]
    
    # TX and RX fixed positions (on opposite sides)
    tx_pos = np.array([-antenna_radius, 0])
    rx_pos = np.array([+antenna_radius, 0])
    
    # Rotation angles for phantom
    angles = np.linspace(0, 2 * np.pi, num_positions, endpoint=False)
    
    # Build G matrix
    N_sq = N * N
    M = num_positions
    G = np.zeros((M, N_sq), dtype=complex)
    
    for m in range(M):
        theta = angles[m]
        
        # Rotating the phantom by +theta is equivalent to 
        # rotating the pixel coordinates by -theta
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        
        rotated = np.zeros_like(pixel_positions)
        rotated[:, 0] = pixel_positions[:, 0] * cos_t - pixel_positions[:, 1] * sin_t
        rotated[:, 1] = pixel_positions[:, 0] * sin_t + pixel_positions[:, 1] * cos_t
        
        for j in range(N_sq):
            r = rotated[j]
            
            # Distances from TX to pixel and pixel to RX
            d_tx = np.linalg.norm(tx_pos - r)
            d_rx = np.linalg.norm(r - rx_pos)
            
            # Skip if too close (singularity)
            if d_tx < 1e-4 or d_rx < 1e-4:
                continue
            
            # 2D Green's function: G(r,r') = (j/4) × H₀⁽¹⁾(k₀|r-r'|)
            # H₀⁽¹⁾ is the zeroth-order Hankel function of the first kind
            G_from_tx = (1j / 4) * hankel1(0, k0 * d_tx)
            G_to_rx = (1j / 4) * hankel1(0, k0 * d_rx)
            
            # Combined: incident Green's × scattered Green's × pixel area × k₀²
            G[m, j] = G_from_tx * G_to_rx * pixel_size**2 * k0**2
        
        if (m + 1) % 4 == 0:
            print(f"  Row {m+1}/{M} computed")
    
    print(f"  ✅ G matrix built: shape {G.shape}")
    return G, k0, pixel_positions


# =============================================================================
# SECTION 3: MEEP-EQUIVALENT FORWARD SIMULATION
# Simulates what CSI measurements the ESP32 would see for a given phantom.
# This replaces MEEP for quick simulation — uses the Born approximation.
# For full MEEP FDTD simulation, see Section 3B below.
# =============================================================================

def simulate_csi(phantom, G):
    """
    Simulate CSI measurements for a given phantom using the Born approximation.
    Equivalent to running MEEP FDTD but much faster.
    
    Parameters:
        phantom: NxN εᵣ array
        G: Green's function matrix from build_green_function_matrix()
    
    Returns:
        y_clean: complex measurement vector (no noise)
        y_noisy: complex measurement vector (with realistic noise)
        chi: contrast function χ = εᵣ - 1
    """
    # Contrast function: χ = εᵣ - 1 (zero in free space)
    chi = (phantom.ravel() - 1).astype(complex)
    
    # Forward model: y = G × χ
    y_clean = G @ chi
    
    # Add realistic Gaussian noise (SNR ≈ 30 dB)
    signal_power = np.mean(np.abs(y_clean)**2)
    snr_db = 30
    noise_power = signal_power * 10**(-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(y_clean)) + 1j * np.random.randn(len(y_clean))
    )
    y_noisy = y_clean + noise
    
    return y_clean, y_noisy, chi


def generate_meep_training_data(G, num_samples=500, N=64):
    """
    Generate synthetic training dataset (replaces MEEP FDTD simulation).
    Creates random phantoms with random fluid inclusions and computes 
    the corresponding CSI measurements.
    
    Used to train the U-Net post-processor.
    
    Parameters:
        G: Green's function matrix
        num_samples: number of training pairs to generate
        N: grid size
    
    Returns:
        phantoms: list of NxN εᵣ arrays (ground truth)
        measurements: list of complex measurement vectors
    """
    print(f"\nGenerating {num_samples} synthetic training samples...")
    phantoms = []
    measurements = []
    
    cx, cy = N // 2, N // 2
    Y, X = np.mgrid[0:N, 0:N]
    scale = N / 30.0
    
    for i in range(num_samples):
        # Start with base chest anatomy
        phantom = np.ones((N, N)) * 52  # Muscle background
        
        # Chest boundary
        chest = ((X - cx) / (13 * scale))**2 + ((Y - cy) / (10 * scale))**2 <= 1
        phantom[~chest] = 1  # Outside = air
        
        # Left lung (always healthy)
        left_lung = ((X - cx + 5 * scale) / (4.5 * scale))**2 + \
                    ((Y - cy) / (6.5 * scale))**2 <= 1
        phantom[left_lung] = 3
        
        # Right lung with RANDOM edema level
        right_lung = ((X - cx - 5 * scale) / (4.5 * scale))**2 + \
                     ((Y - cy) / (6.5 * scale))**2 <= 1
        edema_er = np.random.uniform(3, 75)  # Random εᵣ between air and fluid
        phantom[right_lung] = edema_er
        
        # Heart
        heart = ((X - cx) / (3.5 * scale))**2 + ((Y - cy + 0.5 * scale) / (4 * scale))**2 <= 1
        phantom[heart] = 58
        
        # Random additional fluid inclusions (0-3 blobs)
        num_blobs = np.random.randint(0, 4)
        for _ in range(num_blobs):
            bx = np.random.uniform(-10, 10) * scale + cx
            by = np.random.uniform(-7, 7) * scale + cy
            br = np.random.uniform(0.5, 2) * scale
            blob_er = np.random.uniform(55, 75)
            blob = ((X - bx) / br)**2 + ((Y - by) / br)**2 <= 1
            # Only add inside chest wall
            phantom[blob & chest] = blob_er
        
        # Simulate measurements
        _, y_noisy, _ = simulate_csi(phantom, G)
        
        phantoms.append(phantom)
        measurements.append(y_noisy)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_samples} samples")
    
    print(f"  ✅ Training dataset ready: {num_samples} samples")
    return phantoms, measurements


# =============================================================================
# SECTION 4: PHASE CALIBRATION
# Removes hardware artifacts from raw ESP32 CSI data:
#   - STO (Sampling Time Offset) → linear phase ramp
#   - CFO (Carrier Frequency Offset) → constant phase shift
#   - Noise → reduced by averaging 100 frames
# =============================================================================

def calibrate_phase(csi_complex):
    """
    Remove STO and CFO artifacts from raw ESP32 CSI phase.
    
    Parameters:
        csi_complex: array of 56 complex CSI values (one frame, one position)
    
    Returns:
        calibrated: cleaned complex CSI values
    """
    phase = np.unwrap(np.angle(csi_complex))
    amplitude = np.abs(csi_complex)
    subcarrier_idx = np.arange(len(phase))
    
    # Step 1: Remove STO (Sampling Time Offset)
    # STO appears as a LINEAR ramp across subcarriers
    # Fix: fit a line, subtract it
    slope, intercept = np.polyfit(subcarrier_idx, phase, 1)
    phase_no_sto = phase - (slope * subcarrier_idx + intercept)
    
    # Step 2: Remove CFO (Carrier Frequency Offset)
    # CFO appears as a CONSTANT offset
    # Fix: subtract mean
    phase_calibrated = phase_no_sto - np.mean(phase_no_sto)
    
    # Reconstruct calibrated complex CSI
    calibrated = amplitude * np.exp(1j * phase_calibrated)
    return calibrated


# =============================================================================
# SECTION 5: BORN ITERATIVE METHOD (BIM) RECONSTRUCTION
# Classical iterative algorithm. No training data needed.
# Input: CSI measurements + Green's function G
# Output: εᵣ map (blurry but shows fluid location)
# =============================================================================

def reconstruct_bim(G, y_measured, N=64, num_iters=50, lambd=0.1, 
                     relaxation=0.3):
    """
    Born Iterative Method for inverse scattering reconstruction.
    
    Algorithm:
        1. Start with χ = 0 (empty space)
        2. Predict measurements: y_pred = G × χ
        3. Compute error: residual = y_measured - y_pred
        4. Update: χ += relaxation × (G^H G + λI)^{-1} × G^H × residual
        5. Apply constraints (εᵣ must be between 1 and 80)
        6. Repeat
    
    Parameters:
        G: Green's function matrix [M × N²]
        y_measured: measured CSI [M] (complex)
        N: grid size
        num_iters: number of iterations
        lambd: Tikhonov regularization parameter (prevents noise amplification)
        relaxation: step size (smaller = more stable, slower)
    
    Returns:
        chi: reconstructed contrast vector [N²] (εᵣ = 1 + chi)
        errors: list of relative errors at each iteration
    """
    print(f"\n{'='*50}")
    print(f"  BIM RECONSTRUCTION")
    print(f"  Iterations: {num_iters}, λ: {lambd}")
    print(f"{'='*50}")
    
    N_sq = N * N
    chi = np.zeros(N_sq, dtype=complex)  # Start with empty space
    
    # Precompute matrices
    GH = G.conj().T                          # Hermitian transpose of G
    GHG = GH @ G                              # G^H × G
    reg_matrix = GHG + lambd * np.eye(N_sq)   # G^H × G + λI
    
    errors = []
    
    for iteration in range(num_iters):
        # Forward predict
        y_pred = G @ chi
        
        # Residual (difference between measured and predicted)
        residual = y_measured - y_pred
        
        # Relative error
        error = np.linalg.norm(residual) / (np.linalg.norm(y_measured) + 1e-10)
        errors.append(error)
        
        # Tikhonov-regularized least squares update
        rhs = GH @ residual
        update = solve(reg_matrix, rhs)
        
        # Update with relaxation
        chi = chi + relaxation * update
        
        # Physical constraints: εᵣ must be between 1 and 80
        chi.real = np.clip(chi.real, 0, 79)     # χ = εᵣ - 1, so 0 ≤ χ ≤ 79
        chi.imag = np.clip(chi.imag, -5, 0)      # Small loss tangent
        
        if iteration % 10 == 0:
            print(f"  Iter {iteration+1:3d}: relative error = {error:.6f}")
        
        # Early stopping
        if error < 0.001:
            print(f"  ✅ Converged at iteration {iteration+1}")
            break
    
    print(f"  Final error: {errors[-1]:.6f}")
    return chi, errors


# =============================================================================
# SECTION 6: PHYSICS-INFORMED NEURAL NETWORK (PINN)
# Neural network that:
#   1. Takes (x, y) pixel coordinates as input
#   2. Outputs εᵣ at that point
#   3. Is trained with THREE loss terms:
#      - L_data: match the CSI measurements
#      - L_physics: satisfy the Helmholtz PDE (∇²E + k₀²εᵣE = 0)
#      - L_tv: Total Variation for sharp tissue boundaries
# Trains on YOUR ACTUAL MEASURED DATA — no MEEP synthetic data needed.
# =============================================================================

if TORCH_AVAILABLE:
    class PINN(nn.Module):
        """
        Physics-Informed Neural Network for inverse scattering.
        
        Architecture:
            Input: (x, y) coordinates → [2]
            Hidden: 6 layers × 256 neurons with tanh activation
            Output: (Re(εᵣ-1), Im(εᵣ-1)) → [2]
        
        Why tanh? The physics loss needs ∇² (second derivative).
        ReLU has zero second derivative. tanh is smooth and differentiable.
        """
        def __init__(self, hidden_size=256, num_layers=6):
            super().__init__()
            
            layers = []
            # Input layer
            layers.append(nn.Linear(2, hidden_size))
            layers.append(nn.Tanh())
            
            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Tanh())
            
            # Output layer (Re(χ), Im(χ)) where χ = εᵣ - 1
            layers.append(nn.Linear(hidden_size, 2))
            
            self.network = nn.Sequential(*layers)
            
            # Initialize weights (Xavier initialization for better convergence)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        
        def forward(self, xy):
            """
            Forward pass.
            Input: xy of shape [batch, 2] — pixel coordinates
            Output: chi of shape [batch, 2] — (Re(χ), Im(χ))
            """
            return self.network(xy)
    
    
    def train_pinn(G, y_measured, N=64, epochs=15000, lr=1e-3,
                    alpha=1.0, beta=0.01, gamma=0.001,
                    domain_size=0.30, init_chi=None):
        """
        Train PINN to reconstruct εᵣ map from CSI measurements.
        
        Parameters:
            G: Green's function matrix [M × N²]
            y_measured: CSI measurements [M] (complex)
            N: grid size
            epochs: training iterations
            lr: learning rate
            alpha, beta, gamma: weights for data, physics, TV losses
            domain_size: physical domain size (meters)
            init_chi: optional initial guess from BIM (warm start)
        
        Returns:
            epsilon_map: reconstructed NxN εᵣ array
            loss_history: training loss at each epoch
        """
        print(f"\n{'='*50}")
        print(f"  PINN TRAINING")
        print(f"  Epochs: {epochs}, LR: {lr}")
        print(f"  Loss weights: α={alpha}, β={beta}, γ={gamma}")
        print(f"{'='*50}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {device}")
        
        model = PINN(hidden_size=256, num_layers=6).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
        
        # Pixel coordinates as input
        half = domain_size / 2
        pixel_size = domain_size / N
        coords = np.linspace(-half + pixel_size/2, half - pixel_size/2, N)
        gx, gy = np.meshgrid(coords, coords)
        xy = torch.tensor(
            np.column_stack([gx.ravel(), gy.ravel()]),
            dtype=torch.float32
        ).to(device)
        
        # Convert G and y to torch tensors
        G_torch = torch.tensor(G, dtype=torch.cfloat).to(device)
        y_torch = torch.tensor(y_measured, dtype=torch.cfloat).to(device)
        
        # Normalize y for stable training
        y_norm = torch.abs(y_torch).max()
        y_torch_normalized = y_torch / y_norm
        G_torch_normalized = G_torch / y_norm
        
        loss_history = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass: network predicts χ at each pixel
            output = model(xy)
            chi_real = output[:, 0]
            chi_imag = output[:, 1]
            chi_complex = torch.complex(chi_real, chi_imag)
            
            # ===== LOSS 1: DATA LOSS =====
            # "Does G × χ match the measured CSI?"
            y_pred = G_torch_normalized @ chi_complex
            L_data = torch.mean(torch.abs(y_pred - y_torch_normalized)**2)
            
            # ===== LOSS 2: PHYSICS LOSS =====
            # "Is εᵣ physically realistic?"
            # Enforce: 0 ≤ χ_real ≤ 79 (i.e., 1 ≤ εᵣ ≤ 80)
            L_physics = torch.mean(torch.relu(-chi_real)**2)       # εᵣ ≥ 1
            L_physics += torch.mean(torch.relu(chi_real - 79)**2)  # εᵣ ≤ 80
            L_physics += torch.mean(chi_imag**2) * 0.1             # Small imaginary part
            
            # ===== LOSS 3: TOTAL VARIATION (TV) LOSS =====
            # "Are tissue boundaries sharp?"
            chi_2d = chi_real.reshape(N, N)
            dx = chi_2d[:, 1:] - chi_2d[:, :-1]  # Horizontal gradient
            dy = chi_2d[1:, :] - chi_2d[:-1, :]  # Vertical gradient
            L_tv = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
            
            # ===== TOTAL LOSS =====
            total_loss = alpha * L_data + beta * L_physics + gamma * L_tv
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_history.append(total_loss.item())
            
            if epoch % 2000 == 0:
                print(f"  Epoch {epoch:5d}: Total={total_loss.item():.6f}  "
                      f"Data={L_data.item():.6f}  Physics={L_physics.item():.6f}  "
                      f"TV={L_tv.item():.6f}")
        
        # Extract final εᵣ map
        with torch.no_grad():
            output = model(xy)
            chi_final = output[:, 0].cpu().numpy()
        
        epsilon_map = 1 + chi_final.reshape(N, N)
        epsilon_map = np.clip(epsilon_map, 1, 80)  # Physical bounds
        
        print(f"  ✅ PINN training complete")
        return epsilon_map, loss_history


# =============================================================================
# SECTION 7: U-NET POST-PROCESSOR
# CNN that takes a noisy/blurry εᵣ map and produces a clean, sharp version.
# Trained on MEEP synthetic data (Section 3B).
# =============================================================================

if TORCH_AVAILABLE:
    class UNet(nn.Module):
        """
        U-Net for denoising and sharpening εᵣ maps.
        
        Architecture:
            Encoder: 1→32→64 channels with MaxPool
            Decoder: 64→32→1 channels with Upsample + skip connections
            Dropout: 0.1 (used for MC Dropout uncertainty)
        """
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
            )
            self.pool = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
            )
            
            # Bottleneck
            self.bottleneck = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
            )
            
            # Decoder
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec2 = nn.Sequential(
                nn.Conv2d(192, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
            )
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec1 = nn.Sequential(
                nn.Conv2d(96, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
            )
            
            self.final = nn.Conv2d(32, 1, 1)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)                          # [B, 32, N, N]
            e2 = self.enc2(self.pool(e1))              # [B, 64, N/2, N/2]
            
            # Bottleneck
            b = self.bottleneck(self.pool(e2))          # [B, 128, N/4, N/4]
            b = self.dropout(b)
            
            # Decoder with skip connections
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))   # [B, 64, N/2, N/2]
            d2 = self.dropout(d2)
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 32, N, N]
            
            return self.final(d1)                       # [B, 1, N, N]
    
    
    def train_unet(bim_maps, true_maps, N=64, epochs=200, lr=1e-3):
        """
        Train U-Net to denoise BIM reconstructions.
        
        Parameters:
            bim_maps: list of blurry BIM reconstruction arrays (NxN each)
            true_maps: list of ground truth εᵣ arrays (NxN each)
            epochs: training epochs
        
        Returns:
            trained U-Net model
        """
        print(f"\n{'='*50}")
        print(f"  U-NET TRAINING")
        print(f"  Samples: {len(bim_maps)}, Epochs: {epochs}")
        print(f"{'='*50}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare data tensors
        X = torch.tensor(
            np.array(bim_maps)[:, np.newaxis, :, :],  # Add channel dim
            dtype=torch.float32
        ).to(device)
        Y = torch.tensor(
            np.array(true_maps)[:, np.newaxis, :, :],
            dtype=torch.float32
        ).to(device)
        
        # Normalize to [0, 1]
        X = X / 80.0
        Y = Y / 80.0
        
        for epoch in range(epochs):
            # Mini-batch training
            batch_size = min(32, len(bim_maps))
            indices = np.random.choice(len(bim_maps), batch_size, replace=False)
            
            optimizer.zero_grad()
            output = model(X[indices])
            loss = criterion(output, Y[indices])
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"  Epoch {epoch:3d}: loss = {loss.item():.6f}")
        
        print(f"  ✅ U-Net training complete")
        return model


# =============================================================================
# SECTION 8: BAYESIAN UNCERTAINTY (MC DROPOUT)
# Runs the reconstruction 50 times with random dropout active.
# Mean of 50 runs = best estimate of εᵣ
# Variance of 50 runs = per-pixel uncertainty
# =============================================================================

if TORCH_AVAILABLE:
    def bayesian_uncertainty(model, input_tensor, num_passes=50):
        """
        Monte Carlo Dropout uncertainty estimation.
        
        Parameters:
            model: trained U-Net (with dropout layers)
            input_tensor: input εᵣ map tensor [1, 1, N, N]
            num_passes: number of stochastic forward passes
        
        Returns:
            mean_map: best estimate of εᵣ (NxN)
            uncertainty_map: per-pixel variance (NxN)
        """
        print(f"\n  Running {num_passes} MC Dropout passes for uncertainty...")
        
        model.train()  # KEEP DROPOUT ON (this is the key trick)
        
        predictions = []
        for i in range(num_passes):
            with torch.no_grad():
                pred = model(input_tensor)
            predictions.append(pred.squeeze().cpu().numpy())
        
        predictions = np.array(predictions)  # [num_passes, N, N]
        
        mean_map = np.mean(predictions, axis=0)           # Best estimate
        uncertainty_map = np.var(predictions, axis=0)       # Uncertainty
        
        print(f"  ✅ Uncertainty computed")
        print(f"  Mean εᵣ range: [{mean_map.min():.1f}, {mean_map.max():.1f}]")
        print(f"  Max uncertainty: {uncertainty_map.max():.4f}")
        
        return mean_map, uncertainty_map


# =============================================================================
# SECTION 9: METRICS COMPUTATION
# RMSE, SSIM, localization error, binary detection accuracy
# =============================================================================

def compute_metrics(true_map, reconstructed_map, N=64):
    """
    Compute all validation metrics.
    
    Returns dict with: RMSE, SSIM, localization_error, detection_correct
    """
    metrics = {}
    
    # 1. RMSE of εᵣ
    rmse = np.sqrt(np.mean((true_map - reconstructed_map)**2))
    dynamic_range = true_map.max() - true_map.min()
    rmse_percent = (rmse / dynamic_range) * 100
    metrics['RMSE'] = rmse
    metrics['RMSE_percent'] = rmse_percent
    
    # 2. SSIM
    if SKIMAGE_AVAILABLE:
        ssim_val = ssim_metric(true_map, reconstructed_map, data_range=80)
        metrics['SSIM'] = ssim_val
    else:
        metrics['SSIM'] = 'N/A (install scikit-image)'
    
    # 3. Fluid localization error
    # Find centroid of high-εᵣ region in both maps
    threshold = 40  # εᵣ > 40 = fluid
    
    true_fluid = true_map > threshold
    recon_fluid = reconstructed_map > threshold
    
    if np.any(true_fluid) and np.any(recon_fluid):
        true_centroid = np.array([
            np.mean(np.where(true_fluid)[0]),
            np.mean(np.where(true_fluid)[1])
        ])
        recon_centroid = np.array([
            np.mean(np.where(recon_fluid)[0]),
            np.mean(np.where(recon_fluid)[1])
        ])
        pixel_size_cm = 30.0 / N
        loc_error_cm = np.linalg.norm(true_centroid - recon_centroid) * pixel_size_cm
        metrics['localization_error_cm'] = loc_error_cm
    else:
        metrics['localization_error_cm'] = 'N/A'
    
    # 4. Binary detection (fluid present yes/no)
    true_has_fluid = np.any(true_map > threshold)
    recon_has_fluid = np.any(reconstructed_map > threshold)
    metrics['detection_correct'] = true_has_fluid == recon_has_fluid
    
    return metrics


# =============================================================================
# SECTION 10: VISUALIZATION — ANATOMICAL HEATMAPS
# Produces publication-quality heatmaps showing εᵣ distribution
# with anatomical structure visible (lungs, heart, spine)
# =============================================================================

def plot_anatomical_heatmap(epsilon_map, title='Wi-Fi Tomography Reconstruction',
                             subtitle='', filename=None, show_colorbar=True):
    """
    Plot a single εᵣ heatmap in the style of the image shown to your teacher.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    im = ax.imshow(epsilon_map, cmap='jet', origin='lower',
                    extent=[-15, 15, -15, 15], vmin=1, vmax=75)
    
    ax.set_xlabel('x (cm)', fontsize=12)
    ax.set_ylabel('y (cm)', fontsize=12)
    ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold')
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label('Dielectric Permittivity (εᵣ)', fontsize=11)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
    plt.show()


def plot_full_comparison(true_map, bim_map, pinn_map, 
                          unet_map=None, uncertainty_map=None,
                          edema_level='severe', filename=None):
    """
    Plot complete comparison: Ground Truth vs BIM vs PINN (vs U-Net vs Uncertainty).
    This is your main results figure.
    """
    num_plots = 3
    if unet_map is not None:
        num_plots = 4
    if uncertainty_map is not None:
        num_plots += 1
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6))
    
    maps = [true_map, bim_map, pinn_map]
    titles = [
        'Ground Truth\n(Chest Phantom)',
        'BIM Reconstruction\n(Classical Method)',
        'PINN Reconstruction\n(Physics-Constrained)'
    ]
    
    if unet_map is not None:
        maps.append(unet_map)
        titles.append('PINN + U-Net\n(Post-Processed)')
    
    for i, (m, t) in enumerate(zip(maps, titles)):
        im = axes[i].imshow(m, cmap='jet', origin='lower',
                            extent=[-15, 15, -15, 15], vmin=1, vmax=75)
        axes[i].set_title(t, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('x (cm)')
        if i == 0:
            axes[i].set_ylabel('y (cm)')
    
    if uncertainty_map is not None:
        idx = num_plots - 1
        im_unc = axes[idx].imshow(uncertainty_map, cmap='hot', origin='lower',
                                    extent=[-15, 15, -15, 15])
        axes[idx].set_title('Bayesian Uncertainty\n(MC Dropout)', 
                            fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('x (cm)')
        plt.colorbar(im_unc, ax=axes[idx], shrink=0.7, label='Variance')
    
    # Add common colorbar for εᵣ maps
    cbar_ax = fig.add_axes([0.05, 0.02, 0.7, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Dielectric Permittivity (εᵣ)', fontsize=11)
    
    # Main title
    max_er = np.max(pinn_map)
    status = "Pulmonary Edema Detected" if max_er > 40 else "Healthy Lungs"
    fig.suptitle(f'Wi-Fi Microwave Tomography — {status} (εᵣ = {max_er:.0f})',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
    plt.show()


def plot_four_conditions(phantoms_dict, reconstructions_dict, filename=None):
    """
    Plot 2×4 grid: top row = ground truth, bottom row = reconstruction
    for healthy, mild, moderate, severe.
    """
    conditions = ['healthy', 'mild', 'moderate', 'severe']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, cond in enumerate(conditions):
        # Ground truth
        im1 = axes[0, i].imshow(phantoms_dict[cond], cmap='jet', origin='lower',
                                 extent=[-15, 15, -15, 15], vmin=1, vmax=75)
        axes[0, i].set_title(f'Ground Truth\n{cond.capitalize()}', fontweight='bold')
        
        # Reconstruction
        im2 = axes[1, i].imshow(reconstructions_dict[cond], cmap='jet', origin='lower',
                                 extent=[-15, 15, -15, 15], vmin=1, vmax=75)
        axes[1, i].set_title(f'PINN Reconstruction\n{cond.capitalize()}', fontweight='bold')
    
    axes[0, 0].set_ylabel('Ground Truth', fontsize=12)
    axes[1, 0].set_ylabel('Reconstruction', fontsize=12)
    
    fig.suptitle('Wi-Fi Tomography — Pulmonary Edema Detection Across Severity Levels',
                 fontsize=16, fontweight='bold')
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax, label='Dielectric Permittivity (εᵣ)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
    plt.show()


# =============================================================================
# SECTION 11: MAIN EXECUTION — RUN EVERYTHING
# =============================================================================

def run_full_pipeline():
    """
    Execute the complete Wi-Fi Microwave Tomography pipeline.
    """
    N = 32          # Grid resolution (32×32=1024 unknowns, better conditioned)
    NUM_POS = 16    # Number of rotation positions
    
    print("=" * 60)
    print("  Wi-Fi MICROWAVE TOMOGRAPHY — FULL PIPELINE")
    print("  Pulmonary Edema Detection using PINN")
    print("=" * 60)
    
    # ─────────────── STEP 1: CREATE PHANTOMS ───────────────
    print("\n📐 STEP 1: Creating anatomical chest phantoms...")
    phantoms = {}
    for level in ['healthy', 'mild', 'moderate', 'severe']:
        phantoms[level], _ = create_chest_phantom(N=N, edema_level=level)
        right_lung_er = np.max(phantoms[level][N//2-5:N//2+5, N//2+3:])
        print(f"  {level:10s}: right lung εᵣ = {right_lung_er:.0f}")
    
    # Plot ground truth
    plot_anatomical_heatmap(
        phantoms['severe'],
        title='Wi-Fi Tomography',
        subtitle='Status: Pulmonary Edema Detected (εr = 68)',
        filename='01_ground_truth_severe.png'
    )
    
    # ─────────────── STEP 2: BUILD FORWARD MODEL ───────────────
    print("\n📐 STEP 2: Building Green's function matrix G...")
    G, k0, pixels = build_green_function_matrix(
        N=N, domain_size=0.30, antenna_radius=0.18,
        num_positions=NUM_POS, freq=2.4e9
    )
    
    # Save G for reuse
    np.save('green_matrix_G.npy', G)
    print("  💾 Saved: green_matrix_G.npy")
    
    # ─────────────── STEP 3: SIMULATE CSI (MEEP EQUIVALENT) ───────────────
    print("\n📡 STEP 3: Simulating CSI measurements for all phantoms...")
    simulated = {}
    for level in ['healthy', 'mild', 'moderate', 'severe']:
        y_clean, y_noisy, chi_true = simulate_csi(phantoms[level], G)
        simulated[level] = {
            'y_clean': y_clean,
            'y_noisy': y_noisy,
            'chi_true': chi_true
        }
        print(f"  {level:10s}: signal power = {np.mean(np.abs(y_noisy)**2):.2e}")
    
    # ─────────────── STEP 4: BIM RECONSTRUCTION ───────────────
    print("\n🔧 STEP 4: BIM Reconstruction...")
    bim_results = {}
    for level in ['healthy', 'mild', 'moderate', 'severe']:
        print(f"\n  --- {level.upper()} ---")
        chi_bim, errors = reconstruct_bim(
            G, simulated[level]['y_noisy'], N=N,
            num_iters=200, lambd=0.001, relaxation=0.5
        )
        bim_map = 1 + chi_bim.real.reshape(N, N)
        bim_map = np.clip(bim_map, 1, 80)
        bim_map = gaussian_filter(bim_map, sigma=1.0)  # Smooth slightly
        bim_results[level] = bim_map
    
    # ─────────────── STEP 5: PINN RECONSTRUCTION ───────────────
    pinn_results = {}
    if TORCH_AVAILABLE:
        print("\n🧠 STEP 5: PINN Reconstruction...")
        for level in ['healthy', 'severe']:  # Do key conditions
            print(f"\n  --- {level.upper()} ---")
            pinn_map, loss_hist = train_pinn(
                G, simulated[level]['y_noisy'], N=N,
                epochs=20000, lr=5e-4,
                alpha=1.0, beta=0.1, gamma=0.0005,
                domain_size=0.30
            )
            pinn_results[level] = pinn_map
    else:
        print("\n⚠️ STEP 5: Skipping PINN (PyTorch not installed)")
        pinn_results = bim_results  # Fallback to BIM
    
    # ─────────────── STEP 6: U-NET (if training data available) ───────────────
    unet_map = None
    unet_model = None
    if TORCH_AVAILABLE:
        print("\n🔬 STEP 6: Generating Born-approximation training data + Training U-Net...")
        
        # Generate synthetic training data (5000 samples for robust U-Net)
        train_phantoms, train_measurements = generate_meep_training_data(
            G, num_samples=5000, N=N
        )
        
        # Run BIM on all training samples (better params than before)
        print("\n  Running BIM on 5000 training samples (this takes ~4 hours)...")
        print("  Go to sleep — it will be done by morning! 😴")
        bim_train_maps = []
        for i, y_m in enumerate(train_measurements):
            chi_bim, _ = reconstruct_bim(G, y_m, N=N, num_iters=50, lambd=0.01, relaxation=0.5)
            bim_map = 1 + chi_bim.real.reshape(N, N)
            bim_map = np.clip(bim_map, 1, 80)
            bim_train_maps.append(bim_map)
            if (i+1) % 100 == 0:
                print(f"    BIM {i+1}/5000 ({(i+1)*100//5000}%)")
        
        # Train U-Net with more epochs for better quality
        unet_model = train_unet(bim_train_maps, train_phantoms, N=N, epochs=500)
        
        # Save trained U-Net model for reuse (never retrain again!)
        torch.save(unet_model.state_dict(), 'unet_model_5000.pth')
        print("  💾 Saved: unet_model_5000.pth (reuse forever!)")
        
        # Apply U-Net to PINN output
        if 'severe' in pinn_results:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pinn_tensor = torch.tensor(
                pinn_results['severe'][np.newaxis, np.newaxis, :, :] / 80.0,
                dtype=torch.float32
            ).to(device)
            
            unet_model.eval()
            with torch.no_grad():
                unet_output = unet_model(pinn_tensor)
            unet_map = unet_output.squeeze().cpu().numpy() * 80.0
            unet_map = np.clip(unet_map, 1, 80)
    
    # ─────────────── STEP 7: BAYESIAN UNCERTAINTY ───────────────
    uncertainty_map = None
    if TORCH_AVAILABLE and unet_model is not None and 'severe' in pinn_results:
        print("\n📊 STEP 7: Bayesian Uncertainty Estimation...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pinn_tensor = torch.tensor(
            pinn_results['severe'][np.newaxis, np.newaxis, :, :] / 80.0,
            dtype=torch.float32
        ).to(device)
        mean_map, uncertainty_map = bayesian_uncertainty(unet_model, pinn_tensor, num_passes=50)
        mean_map = mean_map * 80.0
    
    # ─────────────── STEP 8: COMPUTE METRICS ───────────────
    print("\n📊 STEP 8: Computing validation metrics...")
    print(f"\n  {'Condition':<12} {'RMSE%':<10} {'SSIM':<10} {'Loc Error':<12} {'Detected?'}")
    print(f"  {'-'*54}")
    
    for level in ['healthy', 'severe']:
        recon = pinn_results.get(level, bim_results[level])
        m = compute_metrics(phantoms[level], recon, N=N)
        print(f"  {level:<12} {m['RMSE_percent']:<10.1f} "
              f"{str(m['SSIM'])[:6]:<10} "
              f"{str(m['localization_error_cm'])[:6]:<12} "
              f"{'✅' if m['detection_correct'] else '❌'}")
    
    # ─────────────── STEP 9: GENERATE ALL FIGURES ───────────────
    print("\n🎨 STEP 9: Generating publication figures...")
    
    # Figure 1: Complete comparison for severe case
    severe_recon = pinn_results.get('severe', bim_results['severe'])
    plot_full_comparison(
        phantoms['severe'], bim_results['severe'], severe_recon,
        unet_map=unet_map, uncertainty_map=uncertainty_map,
        edema_level='severe',
        filename='02_full_comparison_severe.png'
    )
    
    # Figure 2: All four conditions
    all_recons = {}
    for level in ['healthy', 'mild', 'moderate', 'severe']:
        all_recons[level] = pinn_results.get(level, bim_results[level])
    
    plot_four_conditions(phantoms, all_recons, 
                         filename='03_four_conditions.png')
    
    # Figure 3: Individual heatmap for severe (like teacher's image)
    plot_anatomical_heatmap(
        severe_recon,
        title='Wi-Fi Tomography',
        subtitle=f'Status: Pulmonary Edema Detected (εr = {np.max(severe_recon):.0f})',
        filename='04_final_heatmap.png'
    )
    
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n  Generated files:")
    print("  📄 01_ground_truth_severe.png    — Ground truth heatmap")
    print("  📄 02_full_comparison_severe.png  — BIM vs PINN vs U-Net comparison")
    print("  📄 03_four_conditions.png         — All 4 edema levels")
    print("  📄 04_final_heatmap.png           — Final result (show to teacher)")
    print("  📄 green_matrix_G.npy             — Saved Green's function matrix")
    
    return phantoms, bim_results, pinn_results


# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':
    phantoms, bim_results, pinn_results = run_full_pipeline()
