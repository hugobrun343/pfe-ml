# Rapport d'experimentation -- Classification 3D par deep learning

## 1. Preprocessing

### 1.1 Protocole

Chaque volume 3D (NIfTI) est decoupe en **16 patches** de taille `256x256x32` (mode MAX, grille 4x4).
Les patches sont initialement sauvegardes au format `.nii.gz`, puis convertis en `.npy` pour accelerer le chargement pendant l'entrainement.

Deux axes d'optimisation ont ete explores :
- **Slice selection** : methode pour choisir le meilleur bloc de 32 coupes dans le volume (intensity, intensity_range, variance, entropy)
- **Normalisation** : transformation des intensites avant extraction (z-score, min-max, robust, intensity_global, minmax_p1p99, minmax_p5p95)

### 1.2 Resultats (ResNet3D-50, validation set)

9 configurations testees, evaluees sur un split train/val unique :

| Slice Selection | Normalisation | Extraction | F1 HEALTHY | F1 DISEASED |
|----------------|---------------|------------|------------|-------------|
| **intensity** | **intensity_global** | MAX 16 | **0.954** | **0.965** |
| intensity | minmax_p5p95 | MAX 16 | 0.915 | 0.935 |
| intensity | minmax_p1p99 | MAX 16 | 0.913 | 0.933 |
| intensity | minmax_p1p99 | TOP_N 4 | 0.901 | 0.927 |
| intensity_range | min-max | MAX 16 | 0.850 | 0.884 |
| intensity_range | z-score | MAX 16 | 0.853 | 0.878 |
| intensity_range | robust | TOP_N 8 | 0.839 | 0.880 |
| variance | z-score | MAX 16 | 0.830 | 0.855 |
| entropy | z-score | MAX 16 | 0.819 | 0.851 |

### 1.3 Conclusions preprocessing

- La **slice selection par intensite** domine largement les methodes variance et entropy (+12 points F1).
- La **normalisation intensity_global** (basee sur les statistiques globales du dataset) est nettement superieure aux normalisations par-patch (z-score, min-max, robust).
- Le mode **MAX (16 patches)** est plus performant que TOP_N (4 ou 8 patches), car il couvre l'integralite du volume sans biais de selection.

**Deux configurations retenues pour la suite :**

| Preprocess | Slice Selection | Normalisation | Specificite | Justification |
|------------|----------------|---------------|-------------|---------------|
| **intensity_global** | intensity | intensity_global | Normalisation basee sur les stats globales du dataset | Meilleure performance globale |
| **minmax_p1p99** | intensity | minmax_p1p99 | Normalisation per-stack (percentiles 1/99) | Tests complementaires |

---

## 2. Comparaison des modeles

### 2.1 Architectures testees

| Modele | Type | Params | Source |
|--------|------|--------|--------|
| ResNet3D-50 | CNN residuel | 46.2M | Custom |
| ResNet3D-101 | CNN residuel | 85.2M | Custom |
| SEResNet3D-50 | CNN + Squeeze-Excitation | 48.7M | Custom |
| SEResNet3D-101 | CNN + Squeeze-Excitation | 90.0M | Custom |
| DenseNet3D-121 | Dense connections | 11.3M | MONAI |
| ConvNeXt3D-Large | CNN modernise | 210M | Custom |
| ViT3D-Base | Vision Transformer | 89.0M | Custom |
| Swin3D-Tiny | Swin Transformer | 9.8M | MONAI |
| Swin3D-Small | Swin Transformer | 38.7M | MONAI |

### 2.2 Resultats -- Train/Val split unique

#### Preprocess minmax_p1p99

| Modele | Best val_f1_mean | Optimizer | Note |
|--------|-----------------|-----------|------|
| SEResNet3D-50 | **0.9602** | Adam | |
| SEResNet3D-101 | 0.9548 | Adam | |
| ResNet3D-101 | 0.9515 | Adam | |
| ResNet3D-50 | 0.9427 | Adam | |

#### Preprocess intensity_global -- CNN

| Modele | Best val_f1_mean | Optimizer | Note |
|--------|-----------------|-----------|------|
| DenseNet3D-121 | **0.9528** | Adam | 11.3M params, meilleur ratio perf/taille |
| SEResNet3D-101 | 0.9395 | Adam | |
| SEResNet3D-50 | 0.9393 | Adam | |
| ResNet3D-101 | 0.9284 | Adam | |
| ResNet3D-50 | 0.9202 | Adam | |

#### Preprocess intensity_global -- Transformers (Adam, lr=0.001)

| Modele | Best val_f1_mean | Note |
|--------|-----------------|------|
| ConvNeXt3D-Large | 0.9485 | Seul transformer a converger |
| Swin3D-Tiny | 0.3571 | **Bloque** -- preds constantes |
| Swin3D-Small | 0.3571 | **Bloque** -- preds constantes |
| ViT3D-Base | 0.3571 | **Bloque** -- preds constantes |

> **Diagnostic** : les transformers (sauf ConvNeXt) se sont bloques a F1=0.357 avec une loss ~0.7 plate.
> Le mecanisme d'attention est sensible aux learning rates eleves.
> ConvNeXt, bien que "modernise", reste un CNN convolutionnel pur et n'est pas affecte.

#### Preprocess intensity_global -- Transformers (AdamW + warmup + cosine annealing)

Correction appliquee : `AdamW` (weight_decay=0.05), `lr=0.0001`, warmup 5 epochs, cosine scheduler, gradient clipping 1.0.

| Modele | Best val_f1_mean | Note |
|--------|-----------------|------|
| ViT3D-Base | 0.9167 | Deblocage reussi |
| Swin3D-Tiny | 0.8913 | Deblocage reussi |
| Swin3D-Small | *en cours* | |
| ConvNeXt3D-Large | *en cours* | |

### 2.3 Synthese comparative (meilleur resultat par modele, intensity_global)

| Rang | Modele | val_f1_mean | Optimizer | Params |
|------|--------|------------|-----------|--------|
| 1 | DenseNet3D-121 | **0.9528** | Adam | 11.3M |
| 2 | ConvNeXt3D-Large | 0.9485 | Adam | 210M |
| 3 | SEResNet3D-101 | 0.9395 | Adam | 90.0M |
| 4 | SEResNet3D-50 | 0.9393 | Adam | 48.7M |
| 5 | ResNet3D-101 | 0.9284 | Adam | 85.2M |
| 6 | ResNet3D-50 | 0.9202 | Adam | 46.2M |
| 7 | ViT3D-Base | 0.9167 | AdamW | 89.0M |
| 8 | Swin3D-Tiny | 0.8913 | AdamW | 9.8M |
| 9 | Swin3D-Small | *en cours* | AdamW | 38.7M |

> **Observation** : DenseNet3D-121 obtient le meilleur F1 en validation avec le moins de parametres (11.3M).
> Les modeles CNN classiques (ResNet, SEResNet) sont robustes et stables avec un optimizer simple (Adam).
> Les transformers necessitent un tuning specifique (AdamW, warmup, scheduler) pour converger.

---

## 3. Cross-validation (5-fold)

### 3.1 Protocole

- **10% holdout test** : 94 stacks reserves, jamais utilises pendant l'entrainement
- **90% restants** : 643 stacks repartis en 5 folds stratifies (age, sexe, region, etc.)
- Chaque fold entraine sur 4/5 des 643 stacks, valide sur 1/5
- Preprocess : **intensity_global**
- Tous les modeles entraines avec **Adam, lr=0.001, batch_size variable**

### 3.2 Resultats -- ResNet3D-50

| Fold | val_f1_mean |
|------|------------|
| 0 | 0.9579 |
| 1 | **0.9647** |
| 2 | 0.9498 |
| 3 | 0.9633 |
| 4 | 0.8955 |
| **Moyenne** | **0.9462** |
| Ecart-type | 0.0273 |

### 3.3 Resultats -- SEResNet3D-50

| Fold | val_f1_mean |
|------|------------|
| 0 | 0.9526 |
| 1 | **0.9581** |
| 2 | 0.9313 |
| 3 | 0.9561 |
| 4 | 0.9265 |
| **Moyenne** | **0.9449** |
| Ecart-type | 0.0143 |

### 3.4 Resultats -- ResNet3D-101

| Fold | val_f1_mean |
|------|------------|
| 0 | 0.9483 |
| 1 | **0.9683** |
| 2 | 0.9430 |
| 3 | *en cours* |
| 4 | *en cours* |

### 3.5 Resultats -- SEResNet3D-101

| Fold | val_f1_mean |
|------|------------|
| 0-4 | *en cours (en queue)* |

### 3.6 Resultats -- DenseNet3D-121

| Fold | val_f1_mean |
|------|------------|
| 0-4 | *a lancer* |

### 3.7 Comparaison (folds termines)

| Modele | F1 mean (moy. 5 folds) | Ecart-type | Meilleur fold | Pire fold |
|--------|------------------------|------------|---------------|-----------|
| ResNet3D-50 | **0.9462** | 0.0273 | Fold 1 (0.9647) | Fold 4 (0.8955) |
| SEResNet3D-50 | 0.9449 | 0.0143 | Fold 1 (0.9581) | Fold 4 (0.9265) |

> **Observation** : le fold 4 est systematiquement le plus faible pour les deux modeles,
> ce qui suggere une distribution de donnees legerement plus difficile dans ce fold.
> Le SEResNet3D-50 a un ecart-type plus faible (0.014 vs 0.027), signe d'une meilleure stabilite.

---

## 4. Test sur holdout (pipeline CV test)

### 4.1 Statut pipeline

Le pipeline de test cross-validation est operationnel (`cross-validation-test/`).
Il charge les 5 checkpoints d'un modele, fait l'inference sur les 94 stacks holdout (1504 patches),
agrege les scores par stack (mean probability), ensemble les 5 modeles (mean probability),
et produit des metriques detaillees (F1, accuracy, AUC, confusion matrix).

### 4.2 TODO

- [ ] Attendre la fin des jobs en queue (`cv-test-resnet3d-50`, `cv-test-seresnet3d-50`)
- [ ] Recuperer les resultats holdout pour ResNet3D-50 et SEResNet3D-50
- [ ] Completer les folds ResNet3D-101 et SEResNet3D-101
- [ ] Lancer le test holdout pour ResNet3D-101 et SEResNet3D-101
- [ ] Comparer les performances holdout (vrais resultats de generalisation)
- [ ] Envisager CV pour DenseNet3D-121 (meilleur F1 en train/val unique)
- [ ] Envisager CV pour les transformers (apres validation des resultats AdamW)
- [ ] Analyser les stacks mal classees (per-stack details dans results.json)
- [ ] Completer les resultats Swin3D-Small et ConvNeXt3D-Large (AdamW, en cours)
