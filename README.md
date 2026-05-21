# 🇺🇸 Analyse des Discours d'Investiture des Présidents US (NLP)

Ce dépôt contient le travail pratique d'analyse textuelle et de data visualisation des discours d'investiture américains.

## 🚀 Structure du Projet
* `main.py` : Acquisition et prétraitement lourd des données avec **spaCy** (Tokenisation, Nettoyage, Lemmatisation, POS Tagging, NER).
* `analyse.py` : Analyse des sentiments (**TextBlob**) et extraction des mots-clés par calcul de score **TF-IDF**.
* `visualisation.py` : Génération des graphiques (Évolution temporelle, WordCloud, Barplot en opposition).
* `[NOM_PRENOM]_Rapport_TP.pdf` : Le livrable contenant l'analyse métier et les interprétations des résultats.

## 📦 Installation & Lancement
1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm