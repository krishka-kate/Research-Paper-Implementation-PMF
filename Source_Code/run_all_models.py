import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LOCAL PMF - ALL MODELS WITH PROBE VALIDATION")
print("="*70)

# ============================================
# PATHS
# ============================================

DATA_PATH = r"D:\Krishu ka folder\nf_prize_dataset\training_set"
PROBE_PATH = r"D:\Krishu ka folder\nf_prize_dataset\probe.txt"
RESULTS_PATH = r"D:\pmf_project\results"

os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"Data path exists: {os.path.exists(DATA_PATH)}")
print(f"Probe path exists: {os.path.exists(PROBE_PATH)}")

# ============================================
# LOAD DATA WITH PROBE VALIDATION
# ============================================

def load_data_with_probe(data_path, probe_path, num_movies=1500, max_users=40000):
    
    print("\nLoading probe.txt...")
    probe_movies = {}
    current_movie = None
    
    with open(probe_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                current_movie = int(line.replace(':', ''))
                probe_movies[current_movie] = []
            elif line and current_movie:
                probe_movies[current_movie].append(int(line))
    
    print(f"Probe contains {sum(len(v) for v in probe_movies.values()):,} pairs")
    
    movie_files = sorted([f for f in os.listdir(data_path) if f.endswith('.txt')])
    movie_files = movie_files[:num_movies]
    
    print(f"\nLoading {len(movie_files)} movies...")
    
    user_ids = set()
    for movie_file in movie_files[:300]:
        movie_id = int(movie_file.replace('.txt', '').replace('mv_', ''))
        file_path = os.path.join(data_path, movie_file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        start = 1 if ':' in lines[0] else 0
        for line in lines[start:start+2000]:
            if line.strip():
                user_id = int(line.split(',')[0])
                user_ids.add(user_id)
                if len(user_ids) >= max_users:
                    break
        if len(user_ids) >= max_users:
            break
    
    user_to_idx = {u: i for i, u in enumerate(list(user_ids)[:max_users])}
    movie_to_idx = {}
    print(f"Selected {len(user_to_idx)} users")
    
    n_users = len(user_to_idx)
    n_movies = len(movie_files)
    R = lil_matrix((n_users, n_movies), dtype=np.float32)
    
    validation_pairs = []
    
    for movie_file in tqdm(movie_files):
        movie_id = int(movie_file.replace('.txt', '').replace('mv_', ''))
        
        if movie_id not in movie_to_idx:
            movie_to_idx[movie_id] = len(movie_to_idx)
        
        movie_idx = movie_to_idx[movie_id]
        file_path = os.path.join(data_path, movie_file)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        start = 1 if ':' in lines[0] else 0
        
        probe_users = set(probe_movies.get(movie_id, []))
        
        for line in lines[start:]:
            if not line.strip():
                continue
            parts = line.split(',')
            user_id = int(parts[0])
            rating = float(parts[1])
            
            if user_id in user_to_idx:
                user_idx = user_to_idx[user_id]
                
                if user_id in probe_users:
                    validation_pairs.append((user_idx, movie_idx, rating))
                else:
                    R[user_idx, movie_idx] = rating
        
        if len(movie_to_idx) % 500 == 0:
            gc.collect()
    
    R_csr = R.tocsr()
    
    print(f"\nTraining: {R_csr.shape[0]:,} users, {R_csr.shape[1]:,} movies, {R_csr.nnz:,} ratings")
    print(f"Validation: {len(validation_pairs):,} pairs")
    
    return R_csr, validation_pairs

# Load data
R, validation_pairs = load_data_with_probe(DATA_PATH, PROBE_PATH, num_movies=1500, max_users=40000)

# ============================================
# PMF1 CLASS (High Regularization)
# ============================================

class PMF1:
    def __init__(self, factors=30, lambda_u=0.01, lambda_v=0.001, learning_rate=0.05):
        self.factors = factors
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lr = learning_rate
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    def fit(self, R, validation_pairs, epochs=50, batch_size=50000, verbose=True):
        n_users, n_movies = R.shape
        
        self.U = np.random.normal(0, 0.01, (n_users, self.factors))
        self.V = np.random.normal(0, 0.01, (n_movies, self.factors))
        
        rows, cols = R.nonzero()
        ratings = R.data
        n_obs = len(ratings)
        ratings_scaled = (ratings - 1) / 4.0
        
        train_rmse = []
        val_rmse = []
        
        val_users = np.array([p[0] for p in validation_pairs])
        val_movies = np.array([p[1] for p in validation_pairs])
        val_ratings = np.array([p[2] for p in validation_pairs])
        
        for epoch in range(epochs):
            epoch_start = time.time()
            idx = np.random.permutation(n_obs)
            
            for batch_start in range(0, min(n_obs, 500000), batch_size):
                batch_end = min(batch_start + batch_size, n_obs)
                batch_idx = idx[batch_start:batch_end]
                
                batch_rows = rows[batch_idx]
                batch_cols = cols[batch_idx]
                batch_ratings = ratings_scaled[batch_idx]
                
                pred = np.sum(self.U[batch_rows] * self.V[batch_cols], axis=1)
                pred_sigmoid = self.sigmoid(pred)
                error = batch_ratings - pred_sigmoid
                grad_factor = error * pred_sigmoid * (1 - pred_sigmoid)
                
                for i, (u, m) in enumerate(zip(batch_rows, batch_cols)):
                    grad_U = -grad_factor[i] * self.V[m] + self.lambda_u * self.U[u]
                    grad_V = -grad_factor[i] * self.U[u] + self.lambda_v * self.V[m]
                    
                    self.U[u] -= self.lr * grad_U
                    self.V[m] -= self.lr * grad_V
            
            pred_val = 1 + 4 * self.sigmoid(np.sum(self.U[val_users] * self.V[val_movies], axis=1))
            val_rmse.append(np.sqrt(np.mean((val_ratings - pred_val)**2)))
            
            sample_size = min(50000, n_obs)
            sample_idx = np.random.choice(n_obs, sample_size, replace=False)
            pred_train = 1 + 4 * self.sigmoid(np.sum(self.U[rows[sample_idx]] * self.V[cols[sample_idx]], axis=1))
            train_rmse.append(np.sqrt(np.mean((ratings[sample_idx] - pred_train)**2)))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"PMF1 Epoch {epoch+1:3d}/{epochs} | Train: {train_rmse[-1]:.4f} | Val: {val_rmse[-1]:.4f} | Time: {time.time()-epoch_start:.1f}s")
        
        return train_rmse, val_rmse

# ============================================
# PMF2 CLASS (Low Regularization)
# ============================================

class PMF2:
    def __init__(self, factors=30, lambda_u=0.001, lambda_v=0.0001, learning_rate=0.05):
        self.factors = factors
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lr = learning_rate
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    def fit(self, R, validation_pairs, epochs=50, batch_size=50000, verbose=True):
        n_users, n_movies = R.shape
        
        self.U = np.random.normal(0, 0.01, (n_users, self.factors))
        self.V = np.random.normal(0, 0.01, (n_movies, self.factors))
        
        rows, cols = R.nonzero()
        ratings = R.data
        n_obs = len(ratings)
        ratings_scaled = (ratings - 1) / 4.0
        
        train_rmse = []
        val_rmse = []
        
        val_users = np.array([p[0] for p in validation_pairs])
        val_movies = np.array([p[1] for p in validation_pairs])
        val_ratings = np.array([p[2] for p in validation_pairs])
        
        for epoch in range(epochs):
            epoch_start = time.time()
            idx = np.random.permutation(n_obs)
            
            for batch_start in range(0, min(n_obs, 500000), batch_size):
                batch_end = min(batch_start + batch_size, n_obs)
                batch_idx = idx[batch_start:batch_end]
                
                batch_rows = rows[batch_idx]
                batch_cols = cols[batch_idx]
                batch_ratings = ratings_scaled[batch_idx]
                
                pred = np.sum(self.U[batch_rows] * self.V[batch_cols], axis=1)
                pred_sigmoid = self.sigmoid(pred)
                error = batch_ratings - pred_sigmoid
                grad_factor = error * pred_sigmoid * (1 - pred_sigmoid)
                
                for i, (u, m) in enumerate(zip(batch_rows, batch_cols)):
                    grad_U = -grad_factor[i] * self.V[m] + self.lambda_u * self.U[u]
                    grad_V = -grad_factor[i] * self.U[u] + self.lambda_v * self.V[m]
                    
                    self.U[u] -= self.lr * grad_U
                    self.V[m] -= self.lr * grad_V
            
            pred_val = 1 + 4 * self.sigmoid(np.sum(self.U[val_users] * self.V[val_movies], axis=1))
            val_rmse.append(np.sqrt(np.mean((val_ratings - pred_val)**2)))
            
            sample_size = min(50000, n_obs)
            sample_idx = np.random.choice(n_obs, sample_size, replace=False)
            pred_train = 1 + 4 * self.sigmoid(np.sum(self.U[rows[sample_idx]] * self.V[cols[sample_idx]], axis=1))
            train_rmse.append(np.sqrt(np.mean((ratings[sample_idx] - pred_train)**2)))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"PMF2 Epoch {epoch+1:3d}/{epochs} | Train: {train_rmse[-1]:.4f} | Val: {val_rmse[-1]:.4f} | Time: {time.time()-epoch_start:.1f}s")
        
        return train_rmse, val_rmse

# ============================================
# ADAPTIVE PMF CLASS
# ============================================

class PMFAdaptive:
    def __init__(self, factors=30, learning_rate=0.05):
        self.factors = factors
        self.lr = learning_rate
        self.lambda_u = 0.01
        self.lambda_v = 0.001
    
    def update_priors(self):
        lambda_u_new = 1.0 / (np.mean(self.U**2) + 0.01)
        lambda_v_new = 1.0 / (np.mean(self.V**2) + 0.01)
        lambda_u_new = np.clip(lambda_u_new, 0.005, 0.03)
        lambda_v_new = np.clip(lambda_v_new, 0.0005, 0.003)
        self.lambda_u = 0.95 * self.lambda_u + 0.05 * lambda_u_new
        self.lambda_v = 0.95 * self.lambda_v + 0.05 * lambda_v_new
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    def fit(self, R, validation_pairs, epochs=50, batch_size=50000, verbose=True):
        n_users, n_movies = R.shape
        
        self.U = np.random.normal(0, 0.01, (n_users, self.factors))
        self.V = np.random.normal(0, 0.01, (n_movies, self.factors))
        
        rows, cols = R.nonzero()
        ratings = R.data
        n_obs = len(ratings)
        ratings_scaled = (ratings - 1) / 4.0
        
        train_rmse = []
        val_rmse = []
        
        val_users = np.array([p[0] for p in validation_pairs])
        val_movies = np.array([p[1] for p in validation_pairs])
        val_ratings = np.array([p[2] for p in validation_pairs])
        
        for epoch in range(epochs):
            if epoch > 0 and epoch % 25 == 0:
                self.update_priors()
            
            epoch_start = time.time()
            idx = np.random.permutation(n_obs)
            
            for batch_start in range(0, min(n_obs, 500000), batch_size):
                batch_end = min(batch_start + batch_size, n_obs)
                batch_idx = idx[batch_start:batch_end]
                
                batch_rows = rows[batch_idx]
                batch_cols = cols[batch_idx]
                batch_ratings = ratings_scaled[batch_idx]
                
                pred = np.sum(self.U[batch_rows] * self.V[batch_cols], axis=1)
                pred_sigmoid = self.sigmoid(pred)
                error = batch_ratings - pred_sigmoid
                grad_factor = error * pred_sigmoid * (1 - pred_sigmoid)
                
                for i, (u, m) in enumerate(zip(batch_rows, batch_cols)):
                    grad_U = -grad_factor[i] * self.V[m] + self.lambda_u * self.U[u]
                    grad_V = -grad_factor[i] * self.U[u] + self.lambda_v * self.V[m]
                    
                    self.U[u] -= self.lr * grad_U
                    self.V[m] -= self.lr * grad_V
            
            pred_val = 1 + 4 * self.sigmoid(np.sum(self.U[val_users] * self.V[val_movies], axis=1))
            val_rmse.append(np.sqrt(np.mean((val_ratings - pred_val)**2)))
            
            sample_size = min(50000, n_obs)
            sample_idx = np.random.choice(n_obs, sample_size, replace=False)
            pred_train = 1 + 4 * self.sigmoid(np.sum(self.U[rows[sample_idx]] * self.V[cols[sample_idx]], axis=1))
            train_rmse.append(np.sqrt(np.mean((ratings[sample_idx] - pred_train)**2)))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Adaptive Epoch {epoch+1:3d}/{epochs} | Train: {train_rmse[-1]:.4f} | Val: {val_rmse[-1]:.4f}")
        
        return train_rmse, val_rmse

# ============================================
# CONSTRAINED PMF CLASS
# ============================================

class PMFConstrained:
    def __init__(self, factors=30, lambda_y=0.001, lambda_v=0.001, lambda_w=0.001, learning_rate=0.02):
        self.factors = factors
        self.lambda_y = lambda_y
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.lr = learning_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
    
    def fit(self, R, validation_pairs, epochs=50, verbose=True):
        n_users, n_movies = R.shape
        
        self.Y = np.random.normal(0, 0.01, (n_users, self.factors))
        self.V = np.random.normal(0, 0.01, (n_movies, self.factors))
        self.W = np.random.normal(0, 0.01, (n_movies, self.factors))
        
        rows, cols = R.nonzero()
        
        user_movie_counts = np.zeros(n_users)
        self.U = self.Y.copy()
        
        for u, m in zip(rows, cols):
            self.U[u] += self.W[m]
            user_movie_counts[u] += 1
        
        for u in range(n_users):
            if user_movie_counts[u] > 0:
                self.U[u] /= user_movie_counts[u]
        
        ratings = R.data
        n_obs = len(ratings)
        ratings_scaled = (ratings - 1) / 4.0
        
        train_rmse = []
        val_rmse = []
        
        val_users = np.array([p[0] for p in validation_pairs])
        val_movies = np.array([p[1] for p in validation_pairs])
        val_ratings = np.array([p[2] for p in validation_pairs])
        
        for epoch in range(epochs):
            epoch_start = time.time()
            idx = np.random.permutation(n_obs)
            
            for i in idx[:min(100000, n_obs)]:
                u, m = rows[i], cols[i]
                
                pred = self.sigmoid(self.U[u].dot(self.V[m]))
                error = ratings_scaled[i] - pred
                grad_factor = error * pred * (1 - pred)
                
                grad_Y = -grad_factor * self.V[m] + self.lambda_y * self.Y[u]
                grad_W = -grad_factor * self.U[u] + self.lambda_w * self.W[m]
                grad_V = -grad_factor * self.U[u] + self.lambda_v * self.V[m]
                
                self.Y[u] -= self.lr * grad_Y
                self.W[m] -= self.lr * grad_W
                self.V[m] -= self.lr * grad_V
            
            self.U = self.Y.copy()
            user_movie_counts = np.zeros(n_users)
            
            for u, m in zip(rows, cols):
                self.U[u] += self.W[m]
                user_movie_counts[u] += 1
            
            for u in range(n_users):
                if user_movie_counts[u] > 0:
                    self.U[u] /= user_movie_counts[u]
            
            pred_val = 1 + 4 * self.sigmoid(np.sum(self.U[val_users] * self.V[val_movies], axis=1))
            val_rmse.append(np.sqrt(np.mean((val_ratings - pred_val)**2)))
            
            sample_size = min(50000, n_obs)
            sample_idx = np.random.choice(n_obs, sample_size, replace=False)
            pred_train = 1 + 4 * self.sigmoid(np.sum(self.U[rows[sample_idx]] * self.V[cols[sample_idx]], axis=1))
            train_rmse.append(np.sqrt(np.mean((ratings[sample_idx] - pred_train)**2)))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Constrained Epoch {epoch+1:3d}/{epochs} | Train: {train_rmse[-1]:.4f} | Val: {val_rmse[-1]:.4f} | Time: {time.time()-epoch_start:.1f}s")
        
        return train_rmse, val_rmse

# ============================================
# RUN ALL MODELS
# ============================================

print("\n" + "="*70)
print("TRAINING PMF1 (High Regularization)")
print("="*70)

pmf1 = PMF1(factors=30, lambda_u=0.01, lambda_v=0.001, learning_rate=0.05)
train1, val1 = pmf1.fit(R, validation_pairs, epochs=50)

print("\n" + "="*70)
print("TRAINING PMF2 (Low Regularization)")
print("="*70)

pmf2 = PMF2(factors=30, lambda_u=0.001, lambda_v=0.0001, learning_rate=0.05)
train2, val2 = pmf2.fit(R, validation_pairs, epochs=50)

print("\n" + "="*70)
print("TRAINING ADAPTIVE PMF")
print("="*70)

adaptive = PMFAdaptive(factors=30, learning_rate=0.05)
train_adapt, val_adapt = adaptive.fit(R, validation_pairs, epochs=50)

print("\n" + "="*70)
print("TRAINING CONSTRAINED PMF")
print("="*70)

constrained = PMFConstrained(factors=30, learning_rate=0.02)
train_con, val_con = constrained.fit(R, validation_pairs, epochs=50)

print("\n" + "="*70)
print("RUNNING SVD")
print("="*70)

k = 30
U, s, Vt = svds(R, k=k)
S = np.diag(s)
pred_svd = np.dot(U, np.dot(S, Vt))
pred_svd = np.clip(pred_svd, 1, 5)

val_users = np.array([p[0] for p in validation_pairs])
val_movies = np.array([p[1] for p in validation_pairs])
val_ratings = np.array([p[2] for p in validation_pairs])

pred_val_svd = pred_svd[val_users, val_movies]
svd_rmse = np.sqrt(np.mean((val_ratings - pred_val_svd)**2))

print(f"SVD Validation RMSE: {svd_rmse:.4f}")

# ============================================
# SAVE RESULTS
# ============================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame({
    'epoch': range(1, len(val1)+1),
    'pmf1_val': val1,
    'pmf2_val': val2,
    'adaptive_val': val_adapt,
    'constrained_val': val_con
})
results_df.to_csv(os.path.join(RESULTS_PATH, 'all_models_results.csv'), index=False)

summary = pd.DataFrame({
    'Model': ['PMF1', 'PMF2', 'Adaptive PMF', 'Constrained PMF', 'SVD'],
    'Best Validation RMSE': [min(val1), min(val2), min(val_adapt), min(val_con), svd_rmse]
})
summary.to_csv(os.path.join(RESULTS_PATH, 'results_summary.csv'), index=False)

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(summary.to_string(index=False))

# Plot
plt.figure(figsize=(12, 6))
epochs = range(1, len(val1)+1)
plt.plot(epochs, val1, 'b-', label='PMF1', linewidth=2)
plt.plot(epochs, val2, 'g-', label='PMF2', linewidth=2)
plt.plot(epochs, val_adapt, 'r--', label='Adaptive', linewidth=2)
plt.plot(epochs, val_con, 'purple', label='Constrained', linewidth=2)
plt.axhline(y=svd_rmse, color='orange', linestyle=':', label=f'SVD = {svd_rmse:.4f}')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('All PMF Models - Local Run with Probe Validation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_PATH, 'all_models_plot.png'), dpi=150)
plt.show()

print(f"\nResults saved to: {RESULTS_PATH}")
print("  - all_models_results.csv")
print("  - results_summary.csv")
print("  - all_models_plot.png")