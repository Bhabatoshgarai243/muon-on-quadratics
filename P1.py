import numpy as np
import matplotlib.pyplot as plt

# 1. Experimental Setup
d, n = 16, 16
iterations = 1200
betas = [1.8, 2.0, 2.2]  # Effective step sizes

# Define the Hessian (H) with logarithmically spaced eigenvalues
# i goes from 0 to 15 to match the paper's 1 to 16 index
eigenvalues = [(1/30) * (30 ** (i / 15)) for i in range(16)]
H = np.diag(eigenvalues)

# Initialization: X_0 = I / 4
X_0 = np.eye(d) / 4

# Loss function definition
def calc_loss(X):
    return 0.5 * np.trace(X.T @ H @ X)

# Spectral update map
def spectral_update(G, alpha):
    # Thin SVD
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    # Apply shaping function h(sigma) = tanh(sigma)
    S_new = np.tanh(S)
    # Reconstruct update direction
    update_matrix = U @ np.diag(S_new) @ Vt
    return alpha * update_matrix

# 2. Run the Optimization Loop
results = {}
initial_loss = calc_loss(X_0)

for beta in betas:
    # Since C_h = 1 and lambda_max = 1, alpha = beta
    alpha = beta 
    X_t = X_0.copy()
    loss_history = [initial_loss]
    
    for _ in range(iterations):
        # Gradient of the quadratic at X_t (since X_* = 0)
        G_t = H @ X_t
        
        # Apply the update
        X_t = X_t - spectral_update(G_t, alpha)
        
        # Record loss
        loss_history.append(calc_loss(X_t))
        
    results[beta] = np.array(loss_history) / initial_loss

# 3. Plotting the Results
plt.figure(figsize=(8, 4))

styles = {
    1.8: ('-', 'C0', r'$\beta=1.8$'),
    2.0: ('--', 'C1', r'$\beta=2$'),
    2.2: (':', 'C2', r'$\beta=2.2$')
}

for beta in betas:
    linestyle, color, label = styles[beta]
    plt.plot(results[beta], linestyle=linestyle, color=color, label=label, linewidth=2)

plt.yscale('log')
plt.ylim(1e-6, 10)
plt.xlim(0, 1200)
plt.xlabel('Iteration')
plt.ylabel(r'$f(X_t)/f(X_0)$')
plt.title('(a) Step-size threshold')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('P1_step_size_threshold.svg', dpi=300, bbox_inches='tight')
plt.show()