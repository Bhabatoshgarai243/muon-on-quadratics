import numpy as np
import matplotlib.pyplot as plt

# 1. Experimental Setup & Seed
d, n_cols = 16, 8
iterations = 1400
alpha = 0.01143446
s = 3.080166  # Fixed scale bound

rng = np.random.default_rng(20260912)

# Generate dense Hessian H with condition number 30
eigenvalues = np.array([(1/30) * (30 ** (i / 15)) for i in range(16)])
A = rng.standard_normal((d, d))
Q, _ = np.linalg.qr(A)
H = Q @ np.diag(eigenvalues) @ Q.T

# Generate initial error X_0
E_0_raw = rng.standard_normal((d, n_cols))
X_0 = E_0_raw / np.linalg.norm(E_0_raw, ord='fro')

def calc_loss(X):
    return 0.5 * np.trace(X.T @ H @ X)

initial_loss = calc_loss(X_0)

# Quintic Newton-Schulz stage composed 5 times
def quintic_ns(Y):
    for _ in range(5):
        YtY = Y.T @ Y
        Y = Y @ (3.4445 * np.eye(n_cols) - 4.7750 * YtY + 2.0315 * (YtY @ YtY))
    return Y

# 2. Run Optimization Loop for all methods
methods = ['Fixed-scale NS', 'Exactly normalized NS', 'Exact polar', 'GD']
losses = {m: [initial_loss] for m in methods}
update_lengths = {m: [] for m in methods}

for method in methods:
    X_t = X_0.copy()
    for _ in range(iterations):
        G_t = H @ X_t
        
        # Determine the update direction based on the method
        if method == 'Fixed-scale NS':
            update_step = alpha * quintic_ns(G_t / s)
        elif method == 'Exactly normalized NS':
            update_step = alpha * quintic_ns(G_t / np.linalg.norm(G_t, ord='fro'))
        elif method == 'Exact polar':
            U, _, Vt = np.linalg.svd(G_t, full_matrices=False)
            update_step = alpha * (U @ Vt)
        elif method == 'GD':
            update_step = alpha * G_t
            
        X_t = X_t - update_step
        
        # Track metrics
        losses[method].append(calc_loss(X_t))
        update_lengths[method].append(np.linalg.norm(update_step, ord='fro'))

# Normalize losses relative to initial loss
for m in methods:
    losses[m] = np.array(losses[m]) / initial_loss

# 3. Plotting the Results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

styles = {
    'Fixed-scale NS': ('-', 'C0'),
    'Exactly normalized NS': ('--', 'C3'),
    'Exact polar': (':', 'C1'),
    'GD': ('-.', 'C2')
}

# Top Subplot: Relative Loss
for m in methods:
    ls, color = styles[m]
    ax1.plot(losses[m], linestyle=ls, color=color, label=m, linewidth=2)

ax1.set_yscale('log')
ax1.set_ylabel(r'$f(X_t)/f(X_0)$')
ax1.set_title('(c) Dense-quadratic loss')
ax1.grid(True, alpha=0.3, linestyle='--')

# Bottom Subplot: Update Lengths
for m in methods:
    ls, color = styles[m]
    ax2.plot(update_lengths[m], linestyle=ls, color=color, label=m, linewidth=2)

ax2.set_yscale('log')
ax2.set_ylabel(r'$||X_{t+1} - X_t||_F$')
ax2.set_xlabel('Iteration')
ax2.set_title('(d) Update length')
ax2.grid(True, alpha=0.3, linestyle='--')

# Single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('P3_fixed_vs_per_step.svg', dpi=300, bbox_inches='tight')
plt.show()