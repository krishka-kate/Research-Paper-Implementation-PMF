# Probabilistic Matrix Factorization for Movie Rating Prediction

**Paper Reference**: Salakhutdinov, R., & Mnih, A. (2008). Probabilistic Matrix Factorization. *NIPS*
**Paper Link**: https://www.cs.toronto.edu/~rsalakhu/mltoronto.html

---

## Project Description

I built a Probabilistic Matrix Factorization (PMF) system to predict user ratings for movies. The system learns latent factors that represent user preferences and movie characteristics from historical rating data, then uses these factors to predict unknown ratings. The model scales linearly with the number of observations and handles large-scale sparse datasets efficiently.

---

## What I Built

### Three Model Variants Implemented

| Variant | Description |
|---------|-------------|
| Basic PMF | Standard matrix factorization with fixed regularization |
| PMF with Adaptive Priors | Automatically adjusts regularization during training |
| Constrained PMF | User features depend on which movies they rated |

### Key Features

- Movie-by-movie data loading to handle large datasets without memory issues
- Sparse matrix representation for efficient storage
- Logistic function to bound predictions between 1 and 5 stars
- Mini-batch Stochastic Gradient Descent with momentum for training
- Validation using held-out `probe.txt` ratings

---

## Dataset Used

### Netflix Prize Dataset

| Aspect | Details |
|--------|---------|
| Source | Netflix Prize Competition (2006) |
| Total Movies | 17,770 |
| Movies Used | 1,500 |
| Total Users | 480,189 |
| Users Sampled | 40,000 |
| Total Ratings | 100,480,507 |
| Ratings Used for Training | 1,824,946 |
| Validation Pairs | 18,993 |
| Data Sparsity | ~97% |

---

## Results

### Final Validation RMSE

| Model | Best Validation RMSE |
|-------|---------------------|
| **PMF2 (Low Regularization)** | **0.9920** |
| PMF1 (High Regularization) | 1.0114 |
| Adaptive PMF | 1.0134 |
| Constrained PMF | 1.0788 |

### Training Progress (Validation RMSE by Epoch)

| Epoch | PMF1 | PMF2 | Adaptive | Constrained |
|-------|------|------|----------|-------------|
| 10 | 1.2566 | 1.2558 | 1.2567 | 1.2547 |
| 20 | 1.1551 | 1.1140 | 1.1572 | 1.1552 |
| 30 | 1.0668 | 1.0397 | 1.0679 | 1.1056 |
| 40 | 1.0310 | 1.0087 | 1.0340 | 1.0876 |
| 50 | 1.0114 | 0.9920 | 1.0134 | 1.0788 |

### Training vs Validation (Final Epoch)

| Model | Training RMSE | Validation RMSE | Gap |
|-------|---------------|-----------------|-----|
| PMF1 | 0.9181 | 1.0114 | 0.0933 |
| PMF2 | 0.8672 | 0.9920 | 0.1248 |
| Adaptive | 0.9205 | 1.0134 | 0.0929 |
| Constrained | 1.0510 | 1.0788 | 0.0278 |

> **Key Finding**: PMF2 (low regularization) outperforms PMF1 (high regularization) by **0.0194 RMSE**.

---

## Project Structure
