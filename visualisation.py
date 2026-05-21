import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration esthétique globale (Seaborn) pour maximiser les points de design
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Chargement des données de l'étape précédente
try:
    df = pd.read_pickle("dataset_analyse.pkl")
    # Extraction de l'année depuis la colonne Date pour faire des graphiques temporels
    df['Year'] = df['Date'].str.extract(r'(\d{4})').astype(int)
except Exception as e:
    print(f"❌ Erreur de chargement : {e}. Assure-toi d'avoir exécuté 'analyse.py'.")
    exit()

# -------------------------------------------------------------
# GRAPHIC 1 : RÉPARTITION ET ÉVOLUTION DES SENTIMENTS (4 points)
# -------------------------------------------------------------
def generer_graphiques_sentiments(df):
    print("\nGénération des graphiques de sentiment...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Camembert pour la répartition des tonalités
    tonalite_counts = df['tonalite'].value_counts()
    ax1.pie(tonalite_counts, labels=tonalite_counts.index, autopct='%1.1f%%', 
            colors=sns.color_palette("pastel")[0:len(tonalite_counts)], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax1.set_title("Répartition de la Tonalité Générale des Discours")
    
    # Courbe d'évolution temporelle de la polarité (CORRIGÉE ICI)
    sns.regplot(data=df, x='Year', y='polarite', ax=ax2, color='b', 
                scatter_kws={'s':40, 'alpha':0.7}, line_kws={'color':'red'})
    
    ax2.set_title("Évolution de la Polarité (Positivité) au fil du Temps")
    ax2.set_xlabel("Année de l'investiture")
    ax2.set_ylabel("Score de Polarité (-1 à 1)")
    
    plt.tight_layout()
    plt.savefig("sentiment_analysis.png", dpi=300)
    plt.close()
    print("✅ Graphique 'sentiment_analysis.png' sauvegardé.")

# -------------------------------------------------------------
# GRAPHIC 2 : NUAGE DE MOTS GLOBAL
# -------------------------------------------------------------
def generer_wordcloud(df):
    print("Génération du Nuage de Mots...")
    # On rassemble tous les mots nettoyés de tous les discours
    tous_les_mots = " ".join([" ".join(tokens) for tokens in df['tokens_nettoyes']])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='viridis', max_words=100, random_state=42).generate(tous_les_mots)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Mots les plus fréquents dans les Discours d'Investiture US", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("wordcloud_global.png", dpi=300)
    plt.close()
    print("✅ Nuage de mots 'wordcloud_global.png' sauvegardé.")

# -------------------------------------------------------------
# GRAPHIC 3 : GRAPHIQUE EN BARRES EN OPPOSITION (TF-IDF)
# -------------------------------------------------------------
def generer_graphique_opposition(df):
    print("Génération du graphique en barres en opposition (TF-IDF)...")
    
    # Comparaison : George Washington (Index 0) et le dernier du dataset (Donald Trump)
    idx_p1 = 0                  
    idx_p2 = len(df) - 1        
    
    nom_p1 = df.iloc[idx_p1]['Name']
    nom_p2 = df.iloc[idx_p2]['Name']
    
    textes_comparaison = [df.iloc[idx_p1]['texte_nettoye_str'], df.iloc[idx_p2]['texte_nettoye_str']]
    vectorizer = TfidfVectorizer(max_features=15)
    tfidf_matrix = vectorizer.fit_transform(textes_comparaison).toarray()
    mots = vectorizer.get_feature_names_out()
    
    differences = tfidf_matrix[0] - tfidf_matrix[1]
    
    indices_tries = np.argsort(differences)
    mots_tries = mots[indices_tries]
    diff_triees = differences[indices_tries]
    
    couleurs = ['#e74c3c' if val < 0 else '#3498db' for val in diff_triees]
    
    plt.figure(figsize=(12, 7))
    bars = plt.barh(mots_tries, diff_triees, color=couleurs, edgecolor='none', height=0.6)
    
    plt.axvline(x=0, color='black', linewidth=1.2, linestyle='--')
    plt.title(f"Graphique en Opposition TF-IDF : {nom_p1} vs {nom_p2}", pad=25)
    plt.xlabel("<-- Spécifique au Président Récent  |  Spécifique à G. Washington -->", labelpad=15)
    
    plt.text(min(diff_triees)*0.7, len(mots_tries)-1, nom_p2, color='#e74c3c', weight='bold', fontsize=12)
    plt.text(max(diff_triees)*0.5, 0, nom_p1, color='#3498db', weight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("opposition_tfidf.png", dpi=300)
    plt.close()
    print("✅ Graphique en opposition 'opposition_tfidf.png' sauvegardé.")

# -------------------------------------------------------------
# EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    generer_graphiques_sentiments(df)
    generer_wordcloud(df)
    generer_graphique_opposition(df)
    print("\n🎉 Toutes les visualisations imposées par le barème ont été générées avec succès !")
    print("Vérifie ton dossier, tu y trouveras 3 nouvelles images prêtes pour ton rapport PDF.")