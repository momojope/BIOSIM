# BioSim - Simulateur de jumeaux numériques patients

**BioSim** est une application web qui permet de créer des jumeaux numériques de patients et de simuler l'évolution de leurs paramètres physiologiques en réponse à différents traitements médicamenteux.

---

![Capture d'écran 2025-04-20 001813](https://github.com/user-attachments/assets/c46e4393-2b3d-47ff-9350-f3dca8cb1bf1)
![Capture d'écran 2025-04-20 001853](https://github.com/user-attachments/assets/f50fe586-5fd8-4038-8673-57fba4000e84)


## 🗂️ Table des matières

- [Description](#description)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modèle mathématique](#modèle-mathématique)
- [Technologies utilisées](#technologies-utilisées)
- [Challenges](#challenges)
- [Perspectives](#perspectives)
- [Auteurs](#auteurs)
- [Licence](#licence)

---

## 🩺 Description

**BioSim** est un outil de simulation médical qui permet de créer des profils de patients virtuels et de prédire leur réponse à différents traitements médicamenteux.  
Grâce à des modèles pharmacocinétiques et pharmacodynamiques (PK/PD), l'application modélise :

- l'absorption,
- la distribution,
- le métabolisme,
- l'élimination des médicaments,  
ainsi que leurs effets sur différents paramètres physiologiques comme la glycémie, la pression artérielle et l’inflammation.

---

## 🚀 Fonctionnalités

### ✅ Création de profils patients personnalisés

- Profils prédéfinis : standard, diabétique, âgé, rénal, inflammatoire
- Paramètres personnalisables : âge, poids, sexe, sensibilité à l’insuline, fonction rénale, etc.

### 💊 Simulation de traitements médicamenteux

- Sélection de médicaments : antidiabétiques, anti-inflammatoires, bêta-bloquants, vasodilatateurs
- Planification des prises
- Modélisation de repas et impact glycémique

### 📊 Analyse comparative de scénarios

- Comparaison côte à côte de traitements
- Visualisation des métriques de santé

### 🫀 Visualisation anatomique

- Représentation interactive des effets sur les organes
- Systèmes visualisés : cardiovasculaire, pancréatique, rénal, hépatique, immunitaire

### 🔐 Gestion des utilisateurs et données

- Authentification sécurisée
- Stockage local (SQLite) des profils et historiques
- Export CSV des résultats

---

## 🏗️ Architecture

- `PatientDigitalTwin` : classe principale de simulation
- `pk_pd_model` : équations différentielles simulant les réponses physiologiques
- Interface responsive via **Streamlit**
- Stockage local des profils utilisateurs et simulations avec **SQLite**

---

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/biosim.git
cd biosim

# Créer un environnement virtuel Python
python -m venv venv
source venv/bin/activate        # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run digital_twin_app.py


🧪 Utilisation
Connexion/Inscription : créer un compte ou se connecter

Gestion des patients : créer ou sélectionner un profil

Simulation simple : configurer un traitement et simuler

Comparaison : évaluer plusieurs stratégies thérapeutiques

Visualisation : explorer les effets sur le corps humain

Historique : consulter et réutiliser des simulations passées



def pk_pd_model(self, t, y, medications=None, meal=0):
    """
    Modèle PK/PD complet avec composantes métaboliques, immunitaires et cardiovasculaires.
    y[0]: glucose
    y[1]: insuline
    y[2]: médicament plasma
    y[3]: médicament tissus
    y[4]: cellules immunitaires
    y[5]: inflammation
    y[6]: fréquence cardiaque
    y[7]: pression artérielle
    """
    # Équations différentielles
    ...
Le modèle gère :

Pharmacocinétique : ADME

Pharmacodynamique : impact sur les paramètres cliniques

Interactions entre médicaments

Effets métaboliques des repas

