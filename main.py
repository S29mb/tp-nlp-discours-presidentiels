import os
import pandas as pd
import spacy

# -------------------------------------------------------------
# 1. ACQUISITION DES DONNÉES (1 point)
# -------------------------------------------------------------
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Le fichier {filepath} est introuvable. "
            "Télécharge-le depuis Kaggle et place-le dans le bon dossier."
        )
    
    # Ajout de l'encodage 'latin1' pour éviter l'erreur de décodage sous Windows
    df = pd.read_csv(filepath, encoding='latin1')
    print(f"✅ Données chargées avec succès : {df.shape[0]} discours trouvés.")
    return df

# -------------------------------------------------------------
# 2. PRÉTRAITEMENT DES DONNÉES (4 points)
# -------------------------------------------------------------
# Chargement du modèle linguistique spaCy pour l'anglais
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Le modèle spaCy 'en_core_web_sm' n'est pas installé. Lance : python -m spacy download en_core_web_sm")

def preprocess_speech(text):
    """
    Gère la tokenisation, le nettoyage, la lemmatisation, le POS tagging et la NER.
    """
    if not isinstance(text, str):
        return [], [], []

    # Nettoyage des espaces blancs multiples
    text = " ".join(text.split())

    # Analyse du texte par spaCy
    doc = nlp(text)

    # 1 & 2. Tokenisation, Nettoyage (Stop words, ponctuation) et Lemmatisation
    cleaned_lemmas = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_.strip()) > 1
    ]

    # 3. Part-of-Speech (POS) Tagging
    pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]

    # 4. Named Entity Recognition (NER)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return cleaned_lemmas, pos_tags, entities

# -------------------------------------------------------------
# PIPELINE PRINCIPAL
# -------------------------------------------------------------
if __name__ == "__main__":
    # Nom exact de ton fichier
    DATA_PATH = "presidential_addresses.csv" 
    
    try:
        # Étape 1 : Chargement
        df = load_data(DATA_PATH)
        
        # Vérification et nettoyage de la colonne 'text'
        text_column = 'text'
        if text_column not in df.columns:
            raise KeyError(f"La colonne '{text_column}' est introuvable. Colonnes présentes : {df.columns.tolist()}")
            
        df = df.dropna(subset=[text_column])
        
        print("\n=== Lancement du prétraitement NLP ===")
        print("Note : Cela prend environ 1 à 2 minutes car il y a 58 longs discours à analyser...")
        
        # Application des fonctions NLP
        res = df[text_column].apply(preprocess_speech)
        
        # Séparation des résultats dans de nouvelles colonnes
        df['tokens_nettoyes'] = [r[0] for r in res]
        df['pos_tags'] = [r[1] for r in res]
        df['entities'] = [r[2] for r in res]
        
        print("\n✅ Prétraitement terminé avec succès !")
        print(df[['Name', 'Inaugural Address', 'tokens_nettoyes']].head(3))
        
        # Sauvegarde du dataset traité dans un fichier binaire (pickle) pour conserver les listes Python
        df.to_pickle("dataset_pretraite.pkl")
        print("\nDonnées sauvegardées dans 'dataset_pretraite.pkl'.")
        print("Tu es prêt pour l'étape suivante : Analyses de Sentiment et TF-IDF !")

    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")