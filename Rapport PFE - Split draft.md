# Split train/test — Section 2.8 (draft rapport)

## 2.8 Split des données

### Principe

Le split est appliqué au niveau des **stacks** (volumes), et non des patches. Tous les patches issus d’un même stack appartiennent au même ensemble (train ou test). Cela évite toute fuite d’information entre les deux ensembles.

---

### Étapes du pipeline de split

1. **Filtrage** — sélection des stacks selon des critères métadonnées
2. **Exclusion** — retrait de stacks spécifiques (artefacts, qualité insuffisante)
3. **Split stratifié** — répartition train/test en respectant les distributions

---

### Critères testés ou utilisables

**Filtrage (métadonnées) :**
- Âge (semaines) : min/max
- Région anatomique (DTA, ATA, SAA)
- Sexe, orientation (D/V), fond génétique
- Pression, axial stretch
- Classe (SAIN / MALADE)

**Exclusion :**  
Liste de stacks par identifiant (ex. artefacts visuels, acquisitions incomplètes).

**Split stratifié :**
- Proportion test : 20 % 
- Clés de stratification : Âge, Sexe, Région, Axial stretch, Pression, Classe, Orientation, Genetic
- Graine aléatoire (seed) pour reproductibilité ; variation de la seed pour produire plusieurs splits et évaluer la robustesse.

---

La sortie est un fichier JSON listant les identifiants de stacks pour train et test, ainsi que les métadonnées du split (filtres, seed, ratios).
