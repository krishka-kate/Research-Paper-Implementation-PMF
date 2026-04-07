# Probabilistic Matrix Factorization for Movie Recommendation

**Replication of Salakhutdinov & Mnih (2008) - Netflix Prize Paper**

## 📄 Paper Reference
Salakhutdinov, R., & Mnih, A. (2008). **Probabilistic Matrix Factorization**. *Neural Information Processing Systems (NIPS)*.

## 📌 Overview
This project implements Probabilistic Matrix Factorization (PMF) for collaborative filtering on the Netflix Prize dataset. The implementation includes three variants from the paper:

1. **Basic PMF** - Standard matrix factorization with Gaussian priors
2. **PMF with Adaptive Priors** - Automatic complexity control
3. **Constrained PMF** - User features depend on movies they've rated

## 🎯 Key Results

| Model | Validation RMSE | Paper's RMSE | Difference |
|-------|----------------|--------------|------------|
| **PMF2 (Low Regularization)** | **0.9920** | 0.9253 | +0.0667 |
| PMF1 (High Regularization) | 1.0114 | 0.9430 | +0.0684 |
| Adaptive PMF | 1.0134 | 0.9197 | +0.0937 |
| Constrained PMF | 1.0788 | 0.9016 | +0.1772 |

### Key Finding
PMF2 (low regularization) outperforms PMF1 (high regularization) - **matches paper's core finding**

## 📊 Dataset Statistics

| Parameter | Value |
|-----------|-------|
| Total movies available | 17,770 |
| Movies used in experiment | 1,500 |
| Users sampled | 40,000 |
| Training ratings | 1,824,946 |
| Validation pairs (probe.txt) | 18,993 |
| Data sparsity | ~97% |

## 🏗️ Project Structure
