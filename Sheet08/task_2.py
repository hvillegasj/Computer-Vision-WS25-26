import numpy as np
import matplotlib.pyplot as plt


def load(path):
    """
    Loading the shapes and returning them
    """
    shapes = np.load(path)

    return shapes.astype(np.float64)

def plotter(shapes, title, mean_shape=None, labels=None, styles=None):
    """Helper method for plots
    """
    plt.figure(figsize=(6,6))
    for i in range(shapes.shape[0]):
        pts = shapes[i]
        
        style = styles[i] if styles is not None else "-"
        label = labels[i] if labels is not None else None
        plt.plot(pts[:, 0], pts[:, 1], "-", alpha=0.4, label=label)
        
    if mean_shape is not None:
        plt.plot(mean_shape[:, 0], mean_shape[:, 1], "k-", alpha=1.0, label="Mean shape")
        
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.25)
    
    if labels is not None or mean_shape is not None:
        plt.legend()
    plt.show()

def shapes_to_vectors(shapes):
    num_shapes, num_landmarks, _ = shapes.shape
    
    return shapes.reshape(num_shapes, 2 * num_landmarks)

def vectors_to_shapes(vectors, num_landmarks):
    vectors = np.asarray(vectors)
    if vectors.ndim == 1:
        return vectors.reshape(num_landmarks, 2)
    return vectors.reshape(vectors.shape[0], num_landmarks, 2)

def plot_shapes_grid(shapes_list, titles, suptitle, cols=4):
    n = len(shapes_list)

    cols = max(1, int(cols))
    rows = int(np.ceil(n / cols))
    rows = max(1, rows)
    
    plt.figure(figsize=(4*cols, 4*rows))
    
    for i, (sh, t) in enumerate(zip(shapes_list, titles), start=1):
        ax = plt.subplot(rows, cols, i)
        ax.plot(sh[:, 0], sh[:, 1], "-")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(t)
        ax.grid(True, alpha=0.25)
        
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

def mse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean((a - b) ** 2))


# =================================== Task 2.1 ===================================
# ================================================================================
def compute_affine_transform(source_pts, target_pts):
    """
    Computing affine transform that maps source points to the target points using least square method
    """
    num_landmarks = source_pts.shape[0]
    
    # Build linear system
    X = np.hstack([source_pts, np.ones((num_landmarks, 1))])
    Y = target_pts
    
    # Solve X @ affine_params_matrix ~= Y
     # lstsq returns affine_params_matrix that minimizes ||X M - Y||_2
    affine_params_matrix, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    linear_transform = affine_params_matrix[:2, :].T
    translation = affine_params_matrix[2, :]
    
    return linear_transform, translation

def apply_affine_transform(pts, linear_transform, translation):
    return (pts @ linear_transform.T) + translation

def normalize_shape(pts):
    """Small normalization step to avoid drifting where we center mean at origin and scale to unit RMS distance
    """
    centroid = pts.mean(axis=0)
    centered_pts = pts - centroid
    
    rms_dist = np.sqrt(np.mean(np.sum(centered_pts**2, axis=1)))
    
    if rms_dist < 1e-12:
        return centered_pts
    
    return centered_pts / rms_dist

def generalized_procrustes_affine(shapes, max_iter = 100, tol = 1e-7):
    """
    Iterative affine GPA:
    1) Initialize mean shape
    2) Align all shapes to current mean
    3) Compute new mean shape
    4) Normalize mean shape
    5) Stop when mean change is small
    """
    num_shapes, num_landmarks, _ = shapes.shape
    
    # Avoiding changing of shapes
    original_shapes = shapes.copy()
    
    # Initialize mean shape using first shape
    mean_shape = normalize_shape(original_shapes[0])
    
    aligned_shapes = np.zeros_like(original_shapes)
    
    for i in range(max_iter):
        # Align each shape to current mean
        for j in range(num_shapes):
            linear_transform, translation = compute_affine_transform(original_shapes[j], mean_shape)
            
            aligned_shapes[j] = apply_affine_transform(original_shapes[j], linear_transform, translation)
            
        # Update mean
        new_mean_shape = aligned_shapes.mean(axis=0)
        new_mean_shape = normalize_shape(new_mean_shape)
        
        # Check convergence
        mean_diff = np.linalg.norm(new_mean_shape - mean_shape)
        mean_shape = new_mean_shape
        
        if mean_diff < tol:
            break
            
    return aligned_shapes, mean_shape

# =================================== Task 2.2.1 ===================================
# ==================================================================================

def compute_pca(shape_vectors):
    """
    Manual PCA
    """
    
    num_shapes, _ = shape_vectors.shape
    
    # Mean shape
    mean_vector = shape_vectors.mean(axis=0)
    
    # Center data
    centered = shape_vectors - mean_vector
    
    # Covariance matrix
    cov = (centered.T @ centered) / (num_shapes - 1)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return mean_vector, eigenvectors, eigenvalues

def choose_num_components_for_energy(eigenvalues, energy_threshold):
    """
    Choose smallest N such that cumulatoive varinace >= threshold
    """
    total_energy = np.sum(eigenvalues)
    cumulative_energy = np.cumsum(eigenvalues)
    energy_ratio = cumulative_energy / (total_energy + 1e-12)
    
    N = int(np.searchsorted(energy_ratio, energy_threshold) + 1)
    
    return N


def visualize_pca_modes(mean_vector, eigenvectors, eigenvalues, num_landmarks, mode_indices, sigmas):
    """
    Visualize PCA modes by varying one phi k at a time
    """
    
    for k in mode_indices:
        u_k = eigenvectors[:, k]
        scale = np.sqrt(max(eigenvalues[k], 0))
        
        shapes_list = []
        titles = []
        
        for s in sigmas:
            phi = s * scale
            x = mean_vector + phi * u_k
            sh = vectors_to_shapes(x, num_landmarks)
            shapes_list.append(sh)
            titles.append(f"{s} σ")
        
    
        plot_shapes_grid(shapes_list, titles, suptitle=f"PCA mode k={k+1}")

# =================================== Task 2.2.2 ===================================
# ==================================================================================

def fit_ppca(shape_vectors, latent_dim):
    """
    Fit PPCA using the closed form solution via eigen decomposition of sample covariance
    """
    num_shapes, num_features = shape_vectors.shape
    
    mean_vector = shape_vectors.mean(axis=0)
    centered = shape_vectors - mean_vector
    
    cov = (centered.T @ centered) / (num_shapes - 1)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Take top-q
    eigenvalues_latent = eigenvalues[:latent_dim]
    U_q = eigenvectors[:, :latent_dim]
    
    if latent_dim < num_features:
        noise_variance_sigma2 = float(np.mean(eigenvalues[latent_dim:]))
    else:
        noise_variance_sigma2 = 0.0
    
    # W = U_q(lambda_q - sigma^2I)*0.5R
    adjusted = np.maximum(eigenvalues_latent - noise_variance_sigma2, 0.0)
    matrix_W = U_q @ np.diag(np.sqrt(adjusted))
    
    return mean_vector, matrix_W, noise_variance_sigma2, U_q, eigenvalues_latent

def visualize_ppca_modes(mean_vector, matrix_W, num_landmarks, mode_indices, sigmas):
    """
    Visualize PPCA modes by varying one latent dimesions at a time
    """
    q = matrix_W.shape[1]
    for k in mode_indices:
        if k >= q:
            continue
        
        shapes_list = []
        titles = []
        for s in sigmas:
            z = np.zeros(q)
            z[k] = s
            x = mean_vector + matrix_W @ z
            sh = vectors_to_shapes(x, num_landmarks)
            shapes_list.append(sh)
            titles.append(f"z{k+1}={s}")
            
        plot_shapes_grid(shapes_list, titles, suptitle=f"PPCA latent dim k={k+1}")
    

# =================================== Task 2.3 =====================================
# ==================================================================================

def pca_infer_and_reconstruct(test_vector, mean_vector, eigenvectors, N):
    U = eigenvectors[:, :N]
    phi = U.T @ (test_vector - mean_vector)
    test_hat = mean_vector + U @ phi
    
    return phi, test_hat

def ppca_infer_and_reconstruct(test_vector, mean_vector, matrix_W, noise_variance_sigma2):
    
    latent_dim = matrix_W.shape[1]

    # Posterior mean of latent variables
    lhs = matrix_W.T @ matrix_W + noise_variance_sigma2 * np.eye(latent_dim)
    rhs = matrix_W.T @ (test_vector - mean_vector)

    latent_coeffs = np.linalg.solve(lhs, rhs)

    # Reconstruction
    x_recon = mean_vector + matrix_W @ latent_coeffs

    return latent_coeffs, x_recon

def main():
    MODE = (0, 1, 2)
    SIGMAS = (-3, -2, -1, 0, 1, 2,3)
    
    shapes = load("hands_train.npy")
    print(f"Loaded shapes : {shapes.shape}")

    # Before plot
    plotter(shapes, "Hand shapes BEFORE")
    
    # Run GPA
    aligned_shapes, mean_shape = generalized_procrustes_affine(shapes, max_iter=150, tol=1e-7)
    
    # After plot
    plotter(aligned_shapes, "Hand shapes AFTER GPA", mean_shape)
    
    ##################################################
    shape_vecs = shapes_to_vectors(aligned_shapes)
    
    mean_vector, eigenvectors, eigenvalues = compute_pca(shape_vecs)
    N = choose_num_components_for_energy(eigenvalues, 0.90)
    print(f"PCA N for 90% energy = {N}")
    
    # Visualizing
    visualize_pca_modes(mean_vector, eigenvectors, eigenvalues, num_landmarks=aligned_shapes.shape[1], mode_indices=MODE, sigmas=SIGMAS)
    
    ##################################################
    mean_vector_ppca, matrix_W_ppca, noise_variance_sigma2_ppca, U_q_ppca, eigenvalues_latent_ppca = fit_ppca(shape_vecs, latent_dim=N)
    print(f"PPCA sigma^2 = {noise_variance_sigma2_ppca}")
    
    visualize_ppca_modes(mean_vector_ppca, matrix_W_ppca, num_landmarks=aligned_shapes.shape[1], mode_indices=MODE, sigmas=SIGMAS)
    
    ##################################################
    test = np.load("hands_test.npy")
    print(f"Loaded test : {test.shape}")
    
    linear_transform, translation = compute_affine_transform(test, mean_shape)
    test_aligned = apply_affine_transform(test, linear_transform, translation)
    
    # Vectorize aligned test shape
    x_test = shapes_to_vectors(test_aligned[None, :, :])[0]
    
    # PCA reconstruction
    pca_coeffs, x_recon_pca = pca_infer_and_reconstruct(x_test, mean_vector, eigenvectors, N=N)
    
    recon_pca = vectors_to_shapes(x_recon_pca, num_landmarks=mean_shape.shape[0])
    mse_pca = mse(x_test, x_recon_pca)
    
    print(f"MSE PCA  = {mse_pca:.8f}")
    
    # --- Visualize original vs reconstructed using your plotter ---
    plotter(
        np.stack([test_aligned, recon_pca], axis=0),
        title=f"PCA Reconstruction (N={N}), MSE={mse_pca:.6f}",
        labels=["Original (aligned)", "PCA reconstruction"],
        styles=["k-", "r--"]
    )
    
    # PPCA reconstruction
    latent_coeffs, x_recon_ppca = ppca_infer_and_reconstruct(x_test, mean_vector=mean_vector_ppca, matrix_W=matrix_W_ppca, noise_variance_sigma2=noise_variance_sigma2_ppca)
    recon_ppca = vectors_to_shapes(x_recon_ppca, num_landmarks=mean_shape.shape[0])
    mse_ppca = mse(x_test, x_recon_ppca)
    
    print(f"MSE PPCA  = {mse_ppca:.8f}")
    # --- Visualize original vs reconstructed using your plotter ---
    plotter(
    np.stack([test_aligned, recon_ppca], axis=0),
    title=f"PPCA Reconstruction (q={N}), MSE={mse_ppca:.6f}",
    labels=["Original (aligned)", "PPCA reconstruction"],
    styles=["k-", "b--"]
)

if __name__ == "__main__":
    main()