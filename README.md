# Phoneme Prediction using Machine Learning

A machine learning solution for predicting phoneme sounds from spectral representations, developed as part of an HSE Kaggle competition. This project implements Linear Discriminant Analysis (LDA) with advanced preprocessing techniques to classify the phoneme "g" based on its acoustic features.

## 📋 Overview

This project tackles phoneme classification using spectral data, inspired by the approach described in *The Elements of Statistical Learning* (Hastie, Tibshirani, and Friedman, pp. 148-149). The goal is to predict the phoneme "g" from 256 spectral features extracted from speech signals.

### Problem Statement

Given spectral representations of speech sounds from multiple speakers, predict whether the phoneme is "g" or not. The challenge involves handling:
- High-dimensional spectral data (256 features)
- Speaker-level variability in acoustic characteristics
- Need for efficient computation (60-second time constraint)

## 🎯 Methodology

### Preprocessing Pipeline

The solution employs a three-stage preprocessing approach:

1. **Speaker-level Normalization**
   - Per-speaker z-score normalization removes speaker-specific level/scale differences
   - Allows the model to focus on phoneme patterns rather than speaker characteristics
   - Implementation: `(x - speaker_mean) / speaker_std`

2. **Global Standardization**
   - StandardScaler applied to normalized features
   - Ensures features are on comparable scales for PCA

3. **Dimensionality Reduction**
   - PCA with 50 components reduces feature space from 256 to 50
   - Retains essential spectral information while improving computational efficiency
   - Addresses overfitting by reducing model complexity

### Model: Linear Discriminant Analysis with Shrinkage

**Why LDA?**
- Theoretically well-suited for Gaussian class-conditional densities
- Demonstrated effectiveness for phoneme classification in ESL
- Computationally efficient (linear complexity)
- Performs well with moderate-dimensional data after PCA

**Shrinkage Regularization:**
- Cross-validated shrinkage parameter selection (tested: 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0)
- Stabilizes covariance matrix estimation
- Reduces overfitting, especially important with limited training data
- Uses 5-fold stratified cross-validation for robust evaluation

## 🛠️ Implementation

### Key Technologies
- **Python 3.x**
- **scikit-learn**: LDA, PCA, StandardScaler, cross-validation
- **pandas**: Data manipulation
- **numpy**: Numerical computations

### Core Code Structure

```python
# 1. Speaker Normalization
def speaker_normalize(df, feature_cols, speaker_col):
    df_norm = df.copy()
    df_norm[feature_cols] = df_norm.groupby(speaker_col)[feature_cols].transform(
        lambda x: (x - x.mean()) / np.maximum(x.std(), 1e-6)
    )
    return df_norm

# 2. PCA Dimensionality Reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_normalized)
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 3. LDA with Shrinkage
best_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=best_shrinkage)
best_model.fit(X_train_pca, y_train)
predictions = best_model.predict(X_test_pca)
```

## 📊 Results

The model successfully:
- Reduces dimensionality from 256 to 50 features while preserving predictive power
- Handles speaker variability through normalization
- Achieves stable predictions through shrinkage regularization
- Runs efficiently within the 60-second time constraint

## 📁 Repository Structure

```
.
├── Dubinkin_Vladislav_Phonemes.ipynb  # Main notebook with implementation and analysis
└── README.md                           # This file
```

## 🔍 Key Insights

1. **Speaker normalization is critical**: Raw spectral features vary significantly across speakers; normalization allows the model to learn phoneme-specific patterns
2. **PCA effectiveness**: Reducing from 256 to 50 dimensions maintains discriminative power while preventing overfitting
3. **Shrinkage regularization**: Essential for stable covariance estimation with limited samples per class
4. **Preprocessing order matters**: Speaker normalization → standardization → PCA provides the best results

## 📚 References

### Primary Literature
1. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer.
   - Section 4.3: Linear Discriminant Analysis (pp. 106-119)
   - Section 5.2.3: Phoneme Recognition Example (pp. 148-149, Fig. 5.5)

2. **Hastie, T., Buja, A., & Tibshirani, R. (1995).** *Penalized Discriminant Analysis.* The Annals of Statistics, 23(1), 73-102.

### Additional Resources
- [HSE DSBA ML Course Materials](https://drive.google.com/drive/folders/1fV-Y5XzZ3h-_SNPbblyjDMX6l3Myrkn7)
- [Standard Score (Z-score) - Wikipedia](https://en.wikipedia.org/wiki/Standard_score)

## 👥 Author

**Vladislav Dubinkin**
- Notebook developed for HSE Kaggle Competition 2026

### Original Course Authors
- Oleg Melnikov ([LinkedIn](https://www.linkedin.com/in/olegmelnikov/))
- Alexey Boldyrev ([HSE Profile](https://www.hse.ru/en/org/persons/223985242/))
- Maksim Karpov ([HSE Profile](https://www.hse.ru/en/staff/mekarpov))
- Saraa Ali ([HSE Profile](https://www.hse.ru/en/staff/sara/))

## 📝 License

This project was developed for educational purposes as part of the HSE DSBA ML course.
