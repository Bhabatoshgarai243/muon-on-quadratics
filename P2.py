import numpy as np
import matplotlib.pyplot as plt

# 1. Experimental Setup
d, n = 16, 16
iterations = 2400
fit_window = 300

# Hessian with logarithmically spaced eigenvalues
eigenvalues = np.array([(1/30) * (30 ** (i / 15)) for i in range(16)])
H = np.diag(eigenvalues)
X_0 = np.eye(d) / 4

# 15 equally spaced effective step sizes from 0.3 to 1.98
betas = np.linspace(0.3, 1.98, 15)

# Shapings: Label -> (shaping_function, slope_at_zero)
shapings = {
    'GD': (lambda s: s, 1),
    r'$\tanh(\sigma)$': (np.tanh, 1),
    r'$\tanh(4\sigma)$': (lambda s: np.tanh(4 * s), 4),
    r'$\sigma / \sqrt{\sigma^2 + 0.25}$': (lambda s: s / np.sqrt(s**2 + 0.25), 2)
}

# Spectral update map
def spectral_update(G, alpha, h_func):
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    return alpha * (U @ np.diag(h_func(S)) @ Vt)

# 2. Run the Optimization Loop
results = {name: [] for name in shapings}

for name, (h_func, a) in shapings.items():
    for beta in betas:
        # Match effective step size by scaling alpha
        alpha = beta / a
        X_t = X_0.copy()
        
        error_norms = []
        
        for t in range(iterations):
            G_t = H @ X_t
            X_t = X_t - spectral_update(G_t, alpha, h_func)
            
            # Record the Frobenius norm of the error for the last 300 steps
            if t >= iterations - fit_window:
                error_norms.append(np.linalg.norm(X_t, ord='fro'))
                
        # 3. Calculate local rate via least-squares fit
        t_vals = np.arange(iterations - fit_window, iterations)
        log_errors = np.log(error_norms)
        slope, _ = np.polyfit(t_vals, log_errors, 1)
        measured_rate = np.exp(slope)
        results[name].append(measured_rate)

# 4. Plotting the Results
plt.figure(figsize=(10, 5))

# Plot theoretical GD curve: max |1 - beta * lambda_i|
beta_dense = np.linspace(0.2, 2.05, 200)
theory_rates = [np.max(np.abs(1 - b * eigenvalues)) for b in beta_dense]
plt.plot(beta_dense, theory_rates, 'k-', linewidth=2, label='Theory')

# Plot measured rates with decreasing sizes so they don't hide each other
markers = ['o', '^', '+', 's']
colors = ['gray', 'green', 'purple', 'coral']
sizes = [140, 90, 50, 20] # Shrinking sizes for each subsequent layer

for (name, rates), marker, color, size in zip(results.items(), markers, colors, sizes):
    plt.scatter(betas, rates, label=name, marker=marker, 
                color=color, s=size, edgecolors='k' if marker != '+' else None, zorder=3)

plt.xlim(0.2, 2.05)
plt.ylim(0.93, 1.01)
plt.xlabel(r'Effective step $\beta = \alpha C_h \lambda_{max}$')
plt.ylabel(r'Error rate $\rho$')
plt.title('(b) Local rate (with nested markers)')
plt.legend(loc='lower left', ncol=2)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('P2_local_rate.svg', dpi=300, bbox_inches='tight')
plt.show()