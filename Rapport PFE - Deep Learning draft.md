# Deep Learning — Section 2.9 (draft rapport)

## 2.9 Classification par deep learning

### Objectif

Le modèle effectue une **classification binaire** (SAIN / MALADE) au niveau du **patch** 256×256×32. Chaque patch est prédit individuellement. 

Les prédictions pourront être **agrégées au niveau du stack** par moyenne des probabilités des patches d’un même volume, puis seuil à 0,5 pour la décision finale. Sur les meilleurs runs, l’agrégation améliore les métriques
---

### Architecture et entraînement (phase actuelle)

**Modèle :** ResNet3D-50 (adaptation 3D de ResNet-50)
- Entrée : patches 256×256×32×3 (H, W, D, 3 canaux)
- Sortie : 1 logit → sigmoïde pour la probabilité SAIN/MALADE

**Entraînement :**
- Loss : BCEWithLogitsLoss
- Optimiseur : Adam (lr 0,001, weight_decay 0,0001)
- 100 epochs, batch 64, early stopping (patience 20, min_delta 0,00001)
- Meilleur modèle retenu selon le F1 de la classe MALADE sur l’ensemble de validation

**Métriques :** F1 par classe, accuracy, precision, recall — à la fois au niveau patch et au niveau stack (après agrégation).

---

### Sortie et agrégation

Pour chaque epoch, le pipeline enregistre dans `training_results.json` :
- **samples** : prédictions par patch (stack_id, position, probability, label)
- **volumes** : agrégation par stack (stack_id, aggregated_probability, predicted_label, n_patches)
- **volume_metrics** : métriques au niveau stack (accuracy, F1 par classe, etc.)

L’agrégation (moyenne des probabilités puis seuil 0,5) est calculée à chaque epoch et permet d’évaluer les performances à l’échelle du volume complet.

---

### Place dans le projet

La démarche est organisée en trois phases successives.

**Phase 1 — Évaluation du preprocessing **  
L’objectif est de fixer une stratégie de préparation des données avant de comparer les architectures. Plusieurs configurations ont été testées sur ResNet3D-50 avec différents splits. Les meilleures configurations servent de base pour la suite.

**Phase 2 — Comparaison d’architectures**  
Une fois le preprocessing stabilisé, trois à quatre modèles alternatifs au ResNet3D-50 seront évalués sur les mêmes données. L’objectif est d’identifier l’architecture la plus adaptée au problème.

**Phase 3 — Optimisation des hyperparamètres**  
Une fois le modèle et le preprocessing retenus, une recherche d’hyperparamètres (learning rate, weight decay, batch size, etc.) permettra d’affiner les performances.
