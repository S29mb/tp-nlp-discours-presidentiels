import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------------------
# 3. ANALYSE DE SENTIMENT ET TONALITÉ (3 points)
# -------------------------------------------------------------
def analyser_sentiments(df):
    """
    Calcule la polarité (positif/négatif) et la subjectivité de chaque discours.
    """
    print("\n=== Analyse des Sentiments en cours... ===")
    
    # TextBlob prend une chaîne de caractères en entrée
    # On reconstruit le texte nettoyé à partir de nos tokens pour éviter les bruits
    df['texte_nettoye_str'] = df['tokens_nettoyes'].apply(lambda x: " ".join(x))
    
    # Calcul des scores
    df['polarite'] = df['texte_nettoye_str'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivite'] = df['texte_nettoye_str'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Détermination de la tonalité générale
    def determiner_tonalite(score):
        if score > 0.15: return "Très Optimiste / Positif"
        elif score > 0.05: return "Modérément Positif"
        else: return "Neutre / Réaliste"
        
    df['tonalite'] = df['polarite'].apply(determiner_tonalite)
    print("✅ Analyse des sentiments terminée.")
    return df

# -------------------------------------------------------------
# 4. OCCURRENCE ET TF-IDF (4 points)
# -------------------------------------------------------------
def calculer_tfidf(df):
    """
    Calcule le TF-IDF pour extraire les mots les plus significatifs de chaque discours.
    """
    print("\n=== Calcul du TF-IDF en cours... ===")
    
    # Initialisation du Vectoriseur TF-IDF (max_features=500 pour ne garder que le top des mots)
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(df['texte_nettoye_str'])
    
    # Récupération des mots (features)
    mots = vectorizer.get_feature_names_out()
    
    # Pour chaque discours, on va chercher le mot ayant le score TF-IDF le plus élevé
    mots_cles_principaux = []
    for i in range(len(df)):
        row = tfidf_matrix.getrow(i).toarray()[0]
        # On trie les indices par score décroissant et on prend le top 5 des mots
        top_indices = row.argsort()[-5:][::-1]
        top_mots = [mots[idx] for idx in top_indices if row[idx] > 0]
        mots_cles_principaux.append(", ".join(top_mots))
        
    df['mots_cles_tfidf'] = mots_cles_principaux
    print("✅ Calcul du TF-IDF terminé.")
    return df

# -------------------------------------------------------------
# PIPELINE D'ANALYSE
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Chargement des données prétraitées à l'étape précédente
        df = pd.read_pickle("dataset_pretraite.pkl")
        
        # Exécution des analyses
        df = analyser_sentiments(df)
        df = calculer_tfidf(df)
        
        # Affichage d'un aperçu des résultats pour vérifier
        print("\n=== Aperçu des Résultats Analysés ===")
        print(df[['Name', 'polarite', 'tonalite', 'mots_cles_tfidf']].head(5))
        
        # Sauvegarde pour l'étape finale de Visualisation
        df.to_pickle("dataset_analyse.pkl")
        print("\nDonnées sauvegardées avec succès dans 'dataset_analyse.pkl'.")
        print("Étape suivante : Création des graphiques et visualisations (Partie 5) !")
        
    except FileNotFoundError:
        print("❌ Erreur : Le fichier 'dataset_pretraite.pkl' est introuvable. Lance d'abord 'main.py'.")
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")