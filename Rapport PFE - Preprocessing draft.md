# Préprocessing — Section 2.7 (draft rapport)

## 2.7 Patch extraction and preprocessing

### Paramètres communs

Tous les pipelines produisent des patches de taille **256 × 256 × 32** (H × W × D) à partir de volumes 1042 × 1042 × D × 3. Les dimensions cibles sont fixées pour des raisons de capacité mémoire et d’alignement avec les architectures 3D utilisées.

---

### Composantes du pipeline

Le preprocessing comporte trois étapes :

1. **Normalisation** — mise à l’échelle des intensités
2. **Extraction de patches** — choix des régions spatiales 256×256 à extraire
3. **Sélection de slices** — choix des 32 slices en profondeur (parmi D variable)

---

### Tout ce qui a été testé

**Sélection de slices :**
- *intensity* : slices à intensité totale maximale
- *intensity_range* : slices avec le plus de pixels dans une plage donnée par canal (ex. [300–1800, 0–1000, 300–2000])
- *variance* : slices à variance maximale
- *entropy* : slices à entropie maximale

**Extraction de patches :**
- *MAX* : grille sans chevauchement, nombre maximal de patches par volume (16 typiquement)
- *TOP_N* : N meilleurs patches selon une carte de score (average pooling), avec N = 4 ou 8

**Normalisation :**
- *z-score* : (x − μ) / σ par patch ou par volume
- *min-max* : (x − min) / (max − min)
- *robust* : (x − médiane) / IQR
- *intensity_global* : min/max par canal sur tous les stacks → clip + min-max [0,1]
- *minmax_p1p99* : percentiles 1–99 par stack et par canal → clip + min-max [0,1]
- *minmax_p5p95* : percentiles 5–95 par stack et par canal → clip + min-max [0,1]

**Options complémentaires (normalisation) :** clipping fixe par canal, remapping [0.1–0.9], etc.

---

### Méthodes les plus prometteuses

Les meilleurs résultats ont été obtenus avec :

- **Sélection de slices** : *intensity* — plus stable que les autres.
- **Extraction** : *MAX* en général ; *TOP_N 4* permet de réduire le nombre de patches tout en conservant des performances correctes.
- **Normalisation** : méthodes basées sur des statistiques précalculées plutôt que z-score/min-max sur le patch seul :
  - *intensity_global* : min/max par canal sur l’ensemble du dataset
  - *minmax_p1p99* et *minmax_p5p95* : percentiles par stack et par canal, plus robustes aux outliers

Les statistiques sont calculées de façon à éviter toute fuite entre train et test (uniquement sur les stacks d’entraînement ou avant split selon la méthode). Le split est appliqué au niveau des **stacks** ; tous les patches d’un même stack appartiennent au même split.

---

### Sortie du pipeline

Le pipeline produit :
- Des patches `.nii.gz` dans un dossier dédié
- Un fichier `patches_info.json` (stack_id, label, position, etc.)
- Un fichier `metadata.json` décrivant la configuration utilisée
