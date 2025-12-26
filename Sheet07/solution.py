import numpy as np
import matplotlib.pyplot as plt

def build_matrices(dt, sp, sm):
    """Builds matrcies for the 2D constat acceleration model

    Args:
        dt : time step
        sp : process noise parameter (given : 0.001)
        sm : measurment noise parameter (given : 0.05)
    """
    # postion = position + v*dt + 1/2*a*dt^2
    # velocity = velocity + a*dt
    # acceleration stays constatn but will be corrected by noise and measurements of noise parameter
    
    Psi = np.array([
        [1, 0 , dt, 0, 0.5 * dt**2, 0 ],
        [0, 1, 0, dt, 0, 0.5 * dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    # Measurement matrix Phi
    Phi = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ], dtype=float)

    # Process noise covariance
    sigma_p = sp * np.eye(6)
    
    # Measurment noise covariance
    sigma_m = sm * np.eye(2)
    
    return Psi, Phi, sigma_p, sigma_m

def kalman_filter(observations, dt=0.1, sp=0.001, sm=0.05, x0=None, P0=None):
    """Implementing the Kalman filter for linear systems
    
    """
    Psi, Phi, sigma_p, sigma_m = build_matrices(dt, sp, sm)
    
    N = observations.shape[0]
    
    # Initialize vector
    if x0 is None:
        x0 = np.array([-10, -150, 1, -2, 0, 0], dtype=float)
    else:
        x0 = np.array(x0, dtype=float)
    
    # Initialize covariance
    if P0 is None:
        P0 = 10.0 * np.eye(6)
    else:
        P0= np.array(P0, dtype=float)
    
    x_pred = np.zeros((N, 6))
    P_pred = np.zeros((N, 6, 6))
    x_filt = np.zeros((N, 6))
    P_filt = np.zeros((N, 6, 6))
    
    # Kalman filter loop
    for i in range(N):
        z = observations[i] # measurement[x_measurement, y_measurement]
        
        # ======== Prediction / Time  update ========
        x_before = Psi @ x0
        P_before = Psi @ P0 @ Psi.T + sigma_p
        
        # Save "before" values for smoothing
        x_pred[i] = x_before
        P_pred[i] = P_before
        
        # ======== Correction / Measurment update ======== 
        valid = ~np.isnan(z)   # valid = [True/False, True/False]

        if not np.any(valid):
            # No measurement available -> keep prediction
            x_after = x_before
            P_after = P_before
        else:
            # Select only available measurement dimensions (Partial update)
            Phi_i = Phi[valid, :]                 
            z_k = z[valid]                       
            sigma_m_k = sigma_m[np.ix_(valid, valid)]  

            # Innovation (residual)
            y = z_k - (Phi_i @ x_before)

            # Innovation covariance
            S = Phi_i @ P_before @ Phi_i.T + sigma_m_k

            # Kalman gain
            K = P_before @ Phi_i.T @ np.linalg.inv(S)

            # Update state and covariance
            x_after = x_before + K @ y
            P_after = (np.eye(6) - K @ Phi_i) @ P_before
            
        
        
        # Filtered estimate
        x_filt[i] = x_after
        P_filt[i] = P_after
        
        # Move forward
        x0, P0 = x_after, P_after
        """if i < 5:
            print(f"\nTime step {i}")
            print("Measurement z:", z)
            print("Predicted position:", (Phi @ x_before))
            print("Filtered position:", x_after[:2])"""

    return x_pred, P_pred, x_filt, P_filt

def lag_smoother(x_pred, P_pred, x_filt, P_filt, dt=0.1, lag=5):
    """ Fixed lag smoothing is a method with which we refine the estimate of the state at time (k-lag) using measurments up to time k.
    """
    
    Psi, _, _, _ = build_matrices(dt, sp=0.001, sm=0.05)
    
    N, dim_x = x_filt.shape
    x_smooth = np.copy(x_filt)
    P_smooth = np.copy(P_filt)
    
    for k in range(N):
        start = max(0, k - lag)
        end = k
        
        xs = np.copy(x_filt[start:end+1])
        Ps = np.copy(P_filt[start:end+1])

        # RTS backward recursion inside the window (end - 1, end - 2, ..., start)
        for t in range((end - start) - 1, -1, -1):
            # Map window index t to global index i
            i = start + t
            
            # RTS gain
            P_i = P_filt[i]
            P_next_pred = P_pred[i+1]
            
            G = P_i @ Psi.T @ np.linalg.inv(P_next_pred)
            
            # Smoother state
            xs[t] = x_filt[i] + G @ (xs[t+1] - x_pred[i+1])
            Ps[t] = P_filt[i] + G @ (Ps[t+1] - P_pred[i+1]) @ G.T 
    
        # After smoothing we take the estimate at time = start
        x_smooth[start] = xs[0]
        P_smooth[start] = Ps[0]
    
    return x_smooth, P_smooth
    
def main():
    # Load observations # 1
    try:
        observations = np.load("data/observations.npy")
    except FileNotFoundError:
        observations = np.load("observations.npy")

    nan_x = np.isnan(observations[:, 0]).sum()
    nan_y = np.isnan(observations[:, 1]).sum()

    print("NaNs in x:", nan_x)
    print("NaNs in y:", nan_y)

    # Running the Kalman filter
    x_pred, P_pred, x_filt, P_filt = kalman_filter(observations, dt=0.1, sp=0.001, sm=0.05)
    
    
    x_smooth, P_smooth = lag_smoother(x_pred, P_pred, x_filt, P_filt,dt=0.1, lag=5)

    
    # Visualization
    # Observations are noisy positions (x,y)
    obs_x = observations[:, 0]
    obs_y = observations[:, 1]

    # Filtered trajectory: take x,y from state
    filt_x = x_filt[:, 0]
    filt_y = x_filt[:, 1]
    
    # Smoothed trajectory: take x,y from smoothed state
    smooth_x = x_smooth[:, 0]
    smooth_y = x_smooth[:, 1]

    plt.figure()
    plt.plot(filt_x, filt_y, "-", linewidth=2, label="Kalman Filter (filtered)")
    plt.plot(smooth_x, smooth_y, "--", linewidth=2, label="Fixed-lag smooth (L=5)")
    plt.plot(obs_x, obs_y, ".", alpha=0.4, label="Observations (noisy)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Tracking: observations vs filtered vs fixed-lag smoothed")
    plt.legend()
    plt.axis("equal")
    plt.show()



if __name__ == "__main__":
    main()
