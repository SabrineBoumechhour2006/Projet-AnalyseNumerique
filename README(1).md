# 📐 Application d'Analyse Numérique

> Application web interactive implémentant les principales méthodes d'analyse numérique — développée dans le cadre d'un projet universitaire en ING-3 SOFT, Département Informatique, USTHB.

---

## ✨ Aperçu

Cette application propose une interface web complète pour visualiser et tester les algorithmes numériques fondamentaux, organisés en **3 axes** :

| Axe | Thème | Méthodes |
|-----|-------|----------|
| **Axe 1** | Équations non-linéaires | Dichotomie, Newton-Raphson, Point fixe |
| **Axe 2** | Systèmes linéaires | Jacobi, Gauss-Seidel, Gauss, LU, Cholesky |
| **Axe 3** | Interpolation & Approximation | Lagrange, Newton, Moindres carrés, Descente de gradient |

---

## 🛠️ Technologies utilisées

- **Backend** : Python 3, Flask
- **Frontend** : HTML, CSS, JavaScript
- **Calcul scientifique** : NumPy, SymPy

---

## 🚀 Lancer le projet

### 1. Cloner le repo
```bash
git clone https://github.com/SabrineBoumechhour2006/Projet-AnalyseNumerique.git
cd Projet-AnalyseNumerique
```

### 2. Installer les dépendances
```bash
pip install flask numpy sympy
```

### 3. Démarrer le serveur
```bash
python app.py
```

### 4. Ouvrir dans le navigateur
```
http://localhost:5000
```

---

## 📁 Structure du projet

```
📦 Projet-AnalyseNumerique
 ┣ 📂 Axe1/              → Méthodes non-linéaires
 ┣ 📂 Axe2/
 ┃ ┣ 📂 methodes_directes/    → Gauss, LU, Cholesky
 ┃ ┗ 📂 methodes_indirectes/  → Jacobi, Gauss-Seidel
 ┣ 📂 Axe3/              → Interpolation & Approximation
 ┣ 📂 Image/             → Captures d'écran
 ┣ 📄 app.py             → Serveur Flask & routes API
 ┣ 📄 utils.py           → Utilitaires communs
 ┣ 📄 numanalyse.html    → Interface web
 └ 📄 Rappor_AnalyseNum.pdf
```

---

## 📸 Captures d'écran

### Interface globale
![Interface](Image/Interface_globale/Capture%20d'écran%202026-05-07%20175705.png)

### Axe 1 — Méthodes non-linéaires
![Axe1](Image/Axe1/compar_algo/Capture%20d'écran%202026-05-07%20175858.png)

### Axe 2 — Systèmes linéaires
![Axe2](Image/Axe2/algo_indirect/Capture%20d'écran%202026-05-07%20203957.png)

### Axe 3 — Interpolation
![Axe3](Image/Axe3/Comparaison%20pol%20newton_lagrange/Capture%20d'écran%202026-05-07%20203714.png)

---

## 👩‍💻 Auteure

**Boumechhour Sabrine** — Étudiante en Informatique (ING-3 SOFT), USTHB Alger

[![GitHub](https://img.shields.io/badge/GitHub-SabrineBoumechhour2006-black?logo=github)](https://github.com/SabrineBoumechhour2006)
