# Projet Deep Learning — EMSI Casablanca 2025-2026

**Etudiant :** EL BELLOAUI Amr  
**Filiere :** Data Science et Intelligence Artificielle — 4eme annee cycle ingenieur  
**Module :** Deep Learning  
**Annee universitaire :** 2025-2026

---

## Idee generale du projet

Ce projet est l'evaluation finale du module Deep Learning. Il couvre trois grandes familles d'architectures de deep learning appliquees a trois types de donnees differents :

- **Partie I — MLP** : classification supervisee sur donnees tabulaires (Breast Cancer Wisconsin)
- **Partie II — CNN** : classification d'images avec reseaux convolutionnels (MNIST)
- **Partie III — RNN / Seq2Seq** : modelisation de sequences et traduction automatique (corpus fra-eng)

Chaque partie suit la meme structure : etude theorique → implementation PyTorch → experimentation comparative → analyse critique → question de synthese sur dataset reel.

L'objectif global est de montrer comment le deep learning adapte ses architectures a la geometrie des donnees : structure tabulaire (MLP), structure spatiale 2D (CNN), structure temporelle sequentielle (RNN/Seq2Seq).

---

## Structure du repository

```
Deep-Learning-EMSI-2025/
│
├── README.md                                         # Ce fichier
│
├── Partie1_MLP.ipynb                                 # MLP PyTorch — Breast Cancer Wisconsin
├── Partie2_CNN.ipynb                                 # CNN LeNet — MNIST
├── Partie3_RNN_Seq2Seq.ipynb                        # RNN, LSTM, GRU, Seq2Seq — fra-eng
│
├── Rapport_Partie1_MLP_PyTorch.docx                 # Rapport scientifique Partie I
├── Rapport_Partie2_CNN_EL_BELLOAUI_Amr.docx        # Rapport scientifique Partie II
├── Rapport_Partie3_RNN_Seq2Seq_EL_BELLOAUI_Amr.docx # Rapport scientifique Partie III
│
├── fra.txt                                           # Corpus paires anglais-francais (Tatoeba)
│
├── data/                                             # Dossier reserve aux donnees supplementaires
├── notebooks/                                        # Dossier reserve aux versions alternatives
└── rapports/                                         # Dossier reserve aux versions alternatives
```

---

## Description des notebooks

### Partie1_MLP.ipynb
**Dataset :** Breast Cancer Wisconsin (569 exemples, 30 features, 2 classes)  
**Objectifs couverts :**
- Construction d'un MLP avec `nn.Sequential` et classe personnalisee `nn.Module`
- Inspection des parametres : `named_parameters()`, `state_dict()`
- Comparaison de 3 strategies d'initialisation : gaussienne, constante, Xavier
- Boucle d'entrainement avec sauvegarde du meilleur modele
- Gestion CPU/GPU avec `device`
- Metriques : accuracy, precision, recall, F1-score, matrice de confusion

**Resultats :** F1-score = 0.9389 | Accuracy = 0.9419 (initialisation Xavier)

---

### Partie2_CNN.ipynb
**Dataset :** MNIST (70 000 images 28x28, 10 classes)  
**Objectifs couverts :**
- Implementation manuelle de la correlation croisee 2D et du pooling (NumPy)
- Verification avec `nn.Conv2d`, `nn.MaxPool2d`, `nn.AvgPool2d`
- Architecture CNN type LeNet (61 706 parametres)
- Etude experimentale : padding, stride, type de pooling, nombre de filtres, conv 1x1
- Visualisation des feature maps (Conv1 et Conv2)
- Comparaison MLP vs CNN sur MNIST

**Resultats :** LeNet accuracy = 0.9909 vs MLP accuracy = 0.9812 (3.8x moins de parametres)

---

### Partie3_RNN_Seq2Seq.ipynb
**Dataset :** Corpus de paires anglais-francais (fra-eng, fichier fra.txt)  
**Objectifs couverts :**
- Modele de langage : factorisation par la regle de chaine, perplexite
- Implementation et comparaison RNN / LSTM / GRU
- BPTT et demonstration experimentale du gradient clipping
- Pipeline de donnees : tokenisation, vocabulaire, tokens speciaux, padding, masquage
- Architecture encodeur-decodeur Seq2Seq avec teacher forcing
- Decodage glouton vs beam search (k=3)
- Evaluation par score BLEU et perplexite

**Resultats :** Seq2Seq PPL train 190 → 23 | PPL val 179 → 38 en 15 epoques

---

## Resultats cles

| Partie | Modele | Metrique principale | Valeur |
|--------|--------|---------------------|--------|
| I — MLP | MLP Xavier (30→64→32→2) | F1-score macro | **0.9389** |
| II — CNN | LeNet (61K params) | Accuracy test | **0.9909** |
| II — CNN | MLP baseline (235K params) | Accuracy test | 0.9812 |
| III — Seq2Seq | GRU enc-dec | PPL val finale | **38.1** |

---

## Technologies utilisees

| Outil | Usage |
|-------|-------|
| Python 3.10+ | Langage principal |
| PyTorch 2.x | Implementation des modeles |
| torchvision | Dataset MNIST |
| scikit-learn | Dataset Breast Cancer, metriques |
| NumPy | Implementations manuelles |
| Matplotlib / Seaborn | Visualisations |

---

## Comment executer les notebooks

1. Ouvrir dans **Google Colab** (recommande) ou Jupyter Notebook local
2. Executer les cellules dans l'ordre de haut en bas (`Runtime → Run all`)
3. Les datasets sont telecharges automatiquement :
   - MNIST via `torchvision.datasets`
   - Breast Cancer via `sklearn.datasets`
   - Corpus fra-eng : le fichier `fra.txt` est fourni dans ce repository
4. GPU recommande pour la Partie III (Seq2Seq)

---

## References

- LeCun et al. (1998). *Gradient-Based Learning Applied to Document Recognition*
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*
- Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder*
- Sutskever et al. (2014). *Sequence to Sequence Learning with Neural Networks*
- Zhang et al. *Dive into Deep Learning* — d2l.ai
