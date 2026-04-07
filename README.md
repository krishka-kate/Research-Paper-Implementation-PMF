# Probabilistic Matrix Factorization for Movie Rating Prediction

## Project Description

I built a **Probabilistic Matrix Factorization (PMF)** system to predict user ratings for movies. The system learns latent factors that represent user preferences and movie characteristics from historical rating data, then uses these factors to predict unknown ratings.

The model is designed for large-scale, sparse datasets and scales linearly with the number of observations.

---

## What I Built

### Three Model Variants Implemented

| Variant | Description |
|---------|-------------|
| **Basic PMF** | Standard matrix factorization with fixed regularization |
| **PMF with Adaptive Priors** | Automatically adjusts regularization during training |
| **Constrained PMF** | User features depend on which movies they rated |

### Key Features

- Movie-by-movie data loading to handle large datasets without memory issues
- Sparse matrix representation for efficient storage
- Logistic function to bound predictions between 1 and 5 stars
- Mini-batch Stochastic Gradient Descent with momentum for training
- Validation using held-out probe.txt ratings

---

## Dataset Used

### Netflix Prize Dataset

| Aspect | Details |
|--------|---------|
| **Source** | Netflix Prize Competition (2006) |
| **Total Movies** | 17,770 |
| **Movies Used** | 1,500 |
| **Total Users** | 480,189 |
| **Users Sampled** | 40,000 |
| **Total Ratings** | 100,480,507 |
| **Ratings Used for Training** | 1,824,946 |
| **Validation Pairs** | 18,993 |
| **Data Sparsity** | ~97% |

### Files Used

| File | Purpose |
|------|---------|
| `training_set/` | 17,770 movie files containing user ratings |
| `probe.txt` | Validation set - indicates which ratings to predict |
| `movie_titles.txt` | Movie ID to title/year mapping |

---

## How It Works

### Step 1: Data Loading

The dataset contains one text file per movie. Each file contains:
- First line: Movie ID
- Following lines: User ID, Rating, Date

The probe.txt file contains user-movie pairs reserved for validation. Ratings in probe.txt are held out; remaining ratings are used for training.

### Step 2: Preprocessing

- Ratings scaled from [1-5] to [0-1] using: `(rating - 1) / 4`
- Users limited to 40,000 for memory efficiency
- Movies limited to 1,500
- Data stored in CSR sparse matrix format

### Step 3: Matrix Factorization

The model learns two matrices:
- **U**: User latent factors (40,000 × 30)
- **V**: Movie latent factors (1,500 × 30)

Prediction formula:
