# Système Intelligent d'Analyse de Prêt Bancaire 🏦

Une application Streamlit moderne et interactive pour l'analyse et la prédiction d'approbation de prêts bancaires utilisant plusieurs modèles de machine learning.

## 🌟 Fonctionnalités

- 📊 Exploration interactive des données
- 🤖 Comparaison de 4 modèles ML (Régression Logistique, Random Forest, XGBoost, SVM)
- 🔮 Prédictions en temps réel
- 📈 Visualisations détaillées et interactives
- 💾 Export des modèles entraînés

## 🚀 Installation

1. Clonez le repository :
```bash
git clone <votre-repo-url>
cd Streamlite
```

2. Créez un environnement virtuel et activez-le :
```bash
python -m venv venv
source venv/bin/activate  # Pour Linux/Mac
venv\Scripts\activate     # Pour Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

1. Lancez l'application :
```bash
streamlit run app.py
```

2. Ouvrez votre navigateur à l'adresse : http://localhost:8501

## 📊 Structure de l'Application

- 🏠 **Accueil** : Présentation générale et fonctionnalités
- 📊 **Exploration des Données** : Analyse exploratoire des données
- ⚖️ **Comparaison des Modèles** : Métriques et performances des différents modèles
- 🔮 **Prédiction** : Interface de prédiction en temps réel
- 📈 **Performance Détaillée** : Analyse approfondie des modèles

## 📁 Structure du Projet

```
Streamlite/
├── app.py                  # Application principale
├── requirements.txt        # Dépendances Python
├── loan_approval_dataset.csv   # Dataset
├── .gitignore             # Fichiers ignorés par Git
└── README.md              # Documentation
```

## ⚙️ Modèles Utilisés

1. 📊 **Régression Logistique**
2. 🌳 **Random Forest**
3. 🚀 **XGBoost**
4. 🎯 **SVM (Support Vector Machine)**

## 🛠️ Technologies Utilisées

- Python 3.8+
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## 🌐 Déploiement

L'application peut être déployée sur Streamlit Cloud :

1. Créez un compte sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez votre repository GitHub
3. Déployez l'application en quelques clics

## 📝 License

MIT License

## 👥 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou un pull request.

## 🤝 Support

Pour toute question ou problème, ouvrez une issue dans le repository.
