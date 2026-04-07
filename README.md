---

## How to Run

### 1. Install Dependencies
```bash
pip install numpy scipy pandas matplotlib tqdm
```

### 2. Prepare Dataset

Download the Netflix Prize dataset and place files in the `Dataset/` folder.

### 3. Run Training
```bash
cd Src
python run_all_models.py
```

### 4. Generate Graphs
```bash
python final_results_and_graphs.py
```

---

## Runtime

| Stage | Time |
|-------|------|
| Data loading | 5–10 minutes |
| PMF1 training | 8–10 minutes |
| PMF2 training | 8–10 minutes |
| Adaptive PMF training | 8–10 minutes |
| Constrained PMF training | 8–10 minutes |
| **Total** | **~45–50 minutes** |

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

Salakhutdinov, R., & Mnih, A. (2008). Probabilistic Matrix Factorization. *Advances in Neural Information Processing Systems (NIPS)*, 20, 1257–1264.  
[Paper Link](https://www.cs.toronto.edu/~rsalakhu/mltoronto.html)

---

## Author

**Krishka Kate**  
B.Tech AI & Data Science — NMIMS Indore
