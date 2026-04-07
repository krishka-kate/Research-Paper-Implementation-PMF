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
- Validation using held-out probe.txt ratings

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

## Methodology

### Stage 1: Data Loading

I downloaded the Netflix Prize dataset which contains 100 million ratings from 480,189 users on 17,770 movies. The dataset is split into 17,770 text files, one per movie, each containing User ID, Rating, and Date. I also used the probe.txt file which contains 1.4 million user-movie pairs reserved for validation with their actual ratings withheld.

### Stage 2: Data Preprocessing

I loaded probe.txt first to identify validation pairs, then loaded movie files one by one to avoid memory crashes on a 16GB RAM system. For each rating, I checked if it existed in probe.txt — if yes, it went to the validation set; if no, it went to the training set. This ensures the model never sees validation ratings during training.

Due to memory limitations, I used only 1,500 movies and 40,000 users. Ratings were scaled from the 1–5 range to 0–1 using the formula: scaled_rating = (rating - 1) / 4. Data was stored in CSR sparse matrix format to efficiently handle the ~97% sparsity.

### Stage 3: Model Implementation

**Basic PMF** learns two matrices: a user latent factor matrix of size 40,000 x 30 and a movie latent factor matrix of size 1,500 x 30. Predicted rating = sigmoid(user_vector dot movie_vector), scaled back to 1–5. L2 regularization prevents overfitting. Two versions were trained: PMF1 with high regularization and PMF2 with low regularization.

**Adaptive Priors PMF** works like Basic PMF but automatically adjusts regularization every 25 epochs based on the magnitude of current vectors, smoothed to prevent sudden jumps.

**Constrained PMF** builds the user vector from two parts: a user-specific offset vector plus the average of movie constraint vectors for all movies that user has rated. This helps generalize for users with very few ratings.

### Stage 4: Training Process

- All vectors initialized with random values from N(0, 0.01)
- Trained for 50 epochs with batch size of 50,000
- Optimizer: SGD with learning rate 0.05 and momentum 0.9
- Ratings shuffled each epoch for randomness
- Validation RMSE computed every 10 epochs on held-out probe.txt pairs

### Stage 5: Evaluation

Used Root Mean Squared Error (RMSE) — the same metric as the Netflix Prize competition. Lower RMSE = better performance. Netflix's own baseline score was 0.9514, meaning any model below this outperforms Netflix's own system.

### Stage 6: Results Collection

After training, validation RMSE values at each epoch were saved to CSV files. Five graphs were generated: a comparison of all four models, a replication of the paper's Figure 2 Left (PMF1, PMF2, Adaptive Priors), a replication of Figure 2 Right (Unconstrained vs Constrained PMF), a bar chart comparing my results to the paper, and learning curves showing training vs validation RMSE per model.

### Stage 7: Analysis

My best model (PMF2) achieved validation RMSE of 0.9920 vs the paper's 0.9253 on the full dataset. The gap of 0.0667 is explained by using a smaller dataset. Importantly, my results confirm the paper's core finding: low regularization (PMF2) outperforms high regularization (PMF1). Validation RMSE decreased consistently across all 50 epochs and the training-validation gap was healthy, showing good generalization without severe overfitting.

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

```
PMF_Netflix_Project/
├── README.md
├── requirements.txt
├── Dataset/
│   ├── training_set/
│   │   ├── mv_0000001.txt
│   │   ├── mv_0000002.txt
│   │   └── ...
│   └── probe.txt
├── Src/
│   ├── run_all_models.py
│   └── final_results_and_graphs.py
├── Results/
│   ├── figure1_all_models.png
│   ├── figure2_left_adaptive.png
│   ├── figure2_right_constrained.png
│   ├── figure3_bar_chart.png
│   ├── figure4_learning_curves.png
│   ├── comparison_table.csv
│   ├── raw_validation_results.csv
│   └── results_summary.csv
└── Documents/
    └── Project_Report.docx
```

---

## How to Run

### 1. Install Dependencies

pip install numpy scipy pandas matplotlib tqdm

### 2. Prepare Dataset

Download the Netflix Prize dataset and place files in the Dataset/ folder.

### 3. Run Training

cd Src
python run_all_models.py

### 4. Generate Graphs

python final_results_and_graphs.py

---

## Runtime

| Stage | Time |
|-------|------|
| Data loading | 5-10 minutes |
| PMF1 training | 8-10 minutes |
| PMF2 training | 8-10 minutes |
| Adaptive PMF training | 8-10 minutes |
| Constrained PMF training | 8-10 minutes |
| **Total** | **~45-50 minutes** |

---

## Hyperparameters Used

| Parameter | PMF1 | PMF2 | Adaptive | Constrained |
|-----------|------|------|----------|-------------|
| Latent factors | 30 | 30 | 30 | 30 |
| Learning rate | 0.05 | 0.05 | 0.05 | 0.02 |
| Momentum | 0.9 | 0.9 | 0.9 | 0.9 |
| Epochs | 50 | 50 | 50 | 50 |
| Batch size | 50,000 | 50,000 | 50,000 | 50,000 |
| λ_u | 0.01 | 0.001 | Auto | 0.001 |
| λ_v | 0.001 | 0.0001 | Auto | 0.001 |

---

## Conclusion

I successfully implemented Probabilistic Matrix Factorization for movie rating prediction on the Netflix Prize dataset. My best model (PMF2) achieved a validation RMSE of **0.9920**. The implementation handles large-scale sparse data efficiently and completes training in under 50 minutes.

---

## Reference

Salakhutdinov, R., & Mnih, A. (2008). Probabilistic Matrix Factorization. Advances in Neural Information Processing Systems (NIPS), 20, 1257-1264.
Paper Link: https://www.cs.toronto.edu/~rsalakhu/mltoronto.html

---

## Author

**Krishka Kate**
B.Tech AI & Data Science — NMIMS Indore
