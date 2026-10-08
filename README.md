# TMDB Analytics Pipeline : De l'Extraction à la Recommandation

## 1. Description du projet et du problème métier
Le projet **TMDB Analytics Pipeline** a pour but d'analyser les données de la célèbre base de données de films The Movie Database (TMDB). Le problème métier central consiste à comprendre ce qui génère de l'engagement (votes, popularité) pour un film, à classifier les films selon leur potentiel d'engagement, à segmenter le catalogue (clustering) et à proposer un système de recommandation basé sur le contenu.

## 2. API TMDB et méthode d'extraction
Les données brutes sont collectées depuis l'API officielle de TMDB. Un script d'extraction (`src.extraction.extract`) télécharge les informations essentielles des films (détails, genres, mots-clés) au format JSON, constituant ainsi la base de travail pour l'analyse.

## 3. Principales variables utilisées
- **Variables quantitatives** : `runtime`, `budget`, `revenue`, `popularity`, `vote_average`, `vote_count`.
- **Variables temporelles** : `release_date`.
- **Variables textuelles et catégorielles** : `title`, `overview` (résumé), `original_language`, `genres`, `keywords`.

## 4. Nettoyage et préparation des données
Le module de nettoyage (`src.cleaning.clean`) assure la qualité des données :
- Suppression des films en double (basé sur `movie_id`) et sans titre.
- Traitement des incohérences : les valeurs à 0 pour le budget, les revenus et la durée sont remplacées par `NaN`.
- Conversion de la date de sortie (`release_date`) en format `datetime`.
- Remplacement des résumés (`overview`) manquants par des chaînes de caractères vides.

## 5. Feature Engineering
Le module `src.features.engineering` crée de nouvelles variables pertinentes pour la modélisation :
- **Temporelles** : `release_year`, `release_month`, `release_decade`.
- **Comptage** : Nombre de genres (`n_genres`) et de mots-clés (`n_keywords`).
- **Catégorisation** : Discrétisation de la durée en `runtime_category` (short, medium, long, very_long).
- **Textuelles** : Longueur du résumé (`overview_word_count`).
- **Binaires** : `is_english` (si la langue originale est l'anglais), `budget_known`, `revenue_known`.

## 6. Traitement TF-IDF
Le texte des résumés (`overview`) est transformé à l'aide de la méthode **TF-IDF** (`src.nlp.tfidf`) pour l'analyse de contenu :
- Nettoyage du texte (minuscules, retrait des caractères spéciaux).
- **Paramètres optimaux** : `max_features=5000`, `ngram_range=(1, 2)` (inclusion des bigrammes pour capter des expressions courtes comme "based on"), et `min_df=2`.

## 7. Modèles de classification utilisés
Le pipeline évalue trois modèles d'apprentissage supervisé pour prédire l'engagement :
1. **Régression Logistique** (`LogisticRegression`)
2. **Forêt Aléatoire** (`RandomForestClassifier`)
3. **Machine à Vecteurs de Support Linéaire** (`LinearSVC`)

## 8. Définition de la variable `high_engagement`
La cible de classification (`high_engagement`) est définie de manière binaire. Un film est considéré à fort engagement si son nombre de votes (`vote_count`) est supérieur ou égal au **75ème percentile** (le top 25% des films les plus votés). 
*Note : Le `vote_count` est strictement exclu des features pour éviter toute fuite de données (data leakage).*

## 9. Validation croisée et GridSearchCV
- **Validation croisée** : Utilisation de `StratifiedKFold` (5 splits) pour maintenir l'équilibre des classes lors de l'évaluation des modèles initiaux.
- **GridSearchCV** : Appliqué sur le meilleur modèle (Random Forest) pour optimiser les hyperparamètres : `n_estimators`, `max_depth` et `min_samples_leaf`, en maximisant le score F1.

## 10. Performances et métriques
Les modèles sont évalués sur plusieurs métriques de classification : **Accuracy, Precision, Recall, F1-Score et ROC AUC**. La matrice de confusion est également générée. 
La comparaison avant/après GridSearchCV permet de quantifier l'amélioration apportée par l'ajustement des hyperparamètres.

## 11. Clustering K-Means et choix du nombre de clusters
Le clustering est réalisé avec l'algorithme **K-Means**, appliqué sur deux espaces différents :
1. **Features structurées** (budget, popularité, runtime, etc.).
2. **Matrice TF-IDF** (contenu textuel des résumés).
Le choix du nombre de clusters (k) se fait en calculant le **Silhouette Score** pour k allant de 2 à 10. Le k offrant le score le plus élevé est retenu.

## 12. Interprétation des clusters
- **Clusters structurés** : Analysés via la moyenne des features (budget, vote moyen, etc.) et le genre le plus représenté au sein de chaque cluster.
- **Clusters TF-IDF** : Interprétés en extrayant les mots/bigrammes (top terms) les plus caractéristiques de chaque groupe.

---

## 13. Installation et configuration de l'API

1. Clonez le dépôt et naviguez dans le dossier.
2. Copiez le fichier d'exemple des variables d'environnement :
   ```bash
   cp .env.example .env
   ```
3. Ouvrez `.env` et ajoutez votre clé API TMDB (`TMDB_API_KEY`).
4. (Pour Airflow) Générez et ajoutez une clé Fernet dans le `.env` :
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

## 14. Exécution du pipeline

**Exécution complète sans Airflow (via Docker) :**
```bash
docker compose run --rm app python -m src.pipeline
```

**Exécution étape par étape (en local) :**
```bash
# 1. Extraction
python -m src.extraction.extract
# 2. Nettoyage
python -m src.cleaning.clean
# 3. Base de données
python -m src.database.mongo
# 4. Feature Engineering et Modélisation
python -m src.features.engineering
```

## 15. Lancer l'application Streamlit

L'interface interactive permet de visualiser les données, lancer des recommandations et tester la classification.
```bash
docker compose up -d --build
```
L'application est ensuite accessible sur : **[http://localhost:8501](http://localhost:8501)**

## 16. Exécuter le projet avec Airflow et Docker

L'environnement Docker gère l'ensemble des services : l'application Streamlit, la base MongoDB (avec Mongo Express) et l'orchestrateur **Apache Airflow**.

Une fois les conteneurs lancés (`docker compose up -d`), les interfaces sont disponibles aux adresses suivantes :
- **Airflow** : [http://localhost:8080](http://localhost:8080) (Identifiants : `admin` / `admin`)
- **Mongo Express** : [http://localhost:8081](http://localhost:8081)

Depuis l'interface Airflow, vous pouvez activer le DAG pour orchestrer automatiquement le pipeline (extraction, nettoyage, modélisation).

## 17. Captures d'écran de l'application Streamlit

*(Insérez ici les vraies captures d'écran de l'application)*
- ![Dashboard Général](docs/streamlit_dashboard.png)
- ![Analyse Exploratoire](docs/streamlit_eda.png)

## 18. Exemples de Classification, Clusters et Recommandations

**Exemple de Classification :**
> Titre : Inception  
> Prédiction : **High Engagement** (Probabilité : 92%)

**Exemple de Cluster (TF-IDF) :**
> Cluster "Science-Fiction / Espace"  
> Top mots-clés : `alien`, `space`, `future`, `earth`

**Exemple de Recommandation :**
> Si vous avez aimé *The Matrix* :
> 1. Inception
> 2. Blade Runner 2049
> 3. Ghost in the Shell
