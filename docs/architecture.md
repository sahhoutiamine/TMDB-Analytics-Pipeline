# Architecture du Projet TMDB Analytics Pipeline

Voici le diagramme d'architecture au format Mermaid illustrant l'organisation générale du projet, le flux de données, le pipeline Machine Learning et l'interaction entre les différents composants (Airflow, MongoDB, Python, Streamlit).

```mermaid
graph TD
    %% Styles des noeuds
    classDef external fill:#f9d0c4,stroke:#333,stroke-width:2px;
    classDef python fill:#d4e6f1,stroke:#333,stroke-width:2px;
    classDef database fill:#fcf3cf,stroke:#333,stroke-width:2px;
    classDef web fill:#d5f5e3,stroke:#333,stroke-width:2px;
    classDef orchestrator fill:#ebdef0,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;

    %% Data Source
    TMDB[TMDB API<br/>Source de données]:::external

    %% Orchestration
    Airflow((Apache Airflow<br/>Orchestration Docker)):::orchestrator

    %% Python Data Pipeline
    subgraph Pipeline de Données Python
        Extract[Extraction<br/>src.extraction]:::python
        Clean[Nettoyage & Structuration<br/>src.cleaning]:::python
    end

    %% Storage
    MongoDB[(MongoDB<br/>Base de données NoSQL)]:::database

    %% Python ML Pipeline
    subgraph Pipeline Machine Learning Python
        FE[Feature Engineering<br/>Dates, Catégories, Comptages]:::python
        NLP[Traitement NLP<br/>TF-IDF sur Overviews]:::python
        
        subgraph Modèles ML
            Classif[Classification<br/>Prédiction d'Engagement]:::python
            Cluster[Clustering K-Means<br/>Structuré & Texte]:::python
            Recom[Recommandation<br/>Filtrage Basé Contenu]:::python
        end
    end

    %% UI
    Streamlit[Application Streamlit<br/>Interface Utilisateur]:::web

    %% Flux de données principal
    TMDB -->|JSON / HTTP| Extract
    Extract --> Clean
    Clean -->|Chargement des données| MongoDB
    
    %% Flux ML
    MongoDB -->|Requêtes Pandas| FE
    MongoDB -->|Requêtes Pandas| NLP

    FE --> Classif
    FE --> Cluster
    FE --> Recom
    NLP --> Cluster
    NLP --> Recom

    %% Connexion à l'interface
    Classif -.->|Déploiement Modèles| Streamlit
    Cluster -.->|Groupes & Segments| Streamlit
    Recom -.->|Suggestions de Films| Streamlit

    %% Liens d'Orchestration (Airflow)
    Airflow -.->|Déclenche & Supervise| Extract
    Airflow -.->|Déclenche & Supervise| Clean
    Airflow -.->|Déclenche & Supervise| FE
    Airflow -.->|Déclenche & Supervise| NLP
```
