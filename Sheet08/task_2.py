import numpy as np
import matplotlib.pyplot as plt


def load(path):
    """
    Loading the shapes and returning them
    """
    shapes = np.load(path)

    return shapes.astype(np.float64)

def plotter(shapes, title, mean_shape=None):
    """Helper method for plots
    """
    plt.figure(figsize=(6,6))
    for i in range(shapes.shape[0]):
        pts = shapes[i]
        plt.plot(pts[:, 0], pts[:, 1], "-", alpha=0.4)
        
    if mean_shape is not None:
        plt.plot(mean_shape[:, 0], mean_shape[:, 1], "k-", alpha=1.0)
        
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.25)
    plt.show()

def shapes_to_vectors(shapes):
    num_shapes, num_landmarks, _ = shapes.shape
    
    return shapes.reshape(num_shapes, 2 * num_landmarks)

def vectors_to_shapes(vectors, num_landmarks):
    
    return vectors.reshape(num_landmarks, 2)


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
    eigenvectors_latet = eigenvectors[:latent_dim]
    
    if latent_dim < num_features:
        noise_variance_sigma2 = float(np.mean(eigenvalues[latent_dim:]))
    else:
        noise_variance_sigma2 = 0.0
    
    # W = U_q(lambda_q - sigma^2I)*0.5R
    adjusted = np.maximum(eigenvalues_latent - noise_variance_sigma2, 0.0)
    matrix_W = eigenvectors_latet @ np.diag(np.sqrt(adjusted))
    
    return mean_vector, matrix_W, noise_variance_sigma2, eigenvectors_latet, eigenvalues_latent

def main():
    shapes = load("hands_train.npy")
    print(f"Loaded shapes : {shapes.shape}")
    
    # Before plot
    plotter(shapes, "Hand shapes BEFORE")
    
    # Run GPA
    aligned_shapes, mean_shape = generalized_procrustes_affine(shapes, max_iter=150, tol=1e-7)
    
    # After plot
    plotter(aligned_shapes, "Hand shapes AFTER GPA", mean_shape)
    

if __name__ == "__main__":
    main()