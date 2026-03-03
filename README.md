# RAG Runtime 

Une copie centralisee du RAG est fournie dans ce dossier.

## Arborescence

```text
rag_runtime/
├── README.md
├── .env.example
├── .dockerignore
├── API/
│   ├── __init__.py
│   ├── main.py
│   ├── paths.py
│   ├── fetch_openagenda.py
│   ├── preprocess_events.py
│   ├── vectorstore_index.py
│   ├── rag_pipeline.py
│   ├── ragas_report.py
│   ├── data_quality_checks.py
│   ├── requirements.txt
│   └── README.md
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .dockerignore
│   └── README.md
├── data/
│   └── data_processed/
│       ├── normandie_1y_data.json
│       ├── events_processed.jsonl
│       ├── events_langchain.jsonl
│       └── filtered_events.json
└── vectorstore_normandie/
    ├── index.faiss
    └── index.pkl
```

## Objectif de chaque dossier

- `API/` : le code Python du pipeline RAG, du CLI et des utilitaires d'evaluation/qualite est regroupe.
- `docker/` : la conteneurisation du runtime est definie (build image + commandes compose).
- `data/data_processed/` : les donnees nettoyees et structurees du RAG sont stockees.
- `vectorstore_normandie/` : l'index FAISS utilise par le retriever est stocke.

## Objectif de chaque script Python (`API/`)

- `main.py` : un point d'entree CLI unique est fourni (`fetch`, `preprocess`, `index`, `ask`, `chat`, `report`, `check`, `pipeline`).
- `paths.py` : les chemins de travail (donnees, index, sorties) sont centralises.
- `fetch_openagenda.py` : la collecte des donnees OpenAgenda (paginee ou export CSV) est geree.
- `preprocess_events.py` : la transformation des donnees brutes vers les formats preprocess et JSONL final est effectuee.
- `vectorstore_index.py` : la creation, le chargement et l'interrogation de l'index FAISS sont geres.
- `rag_pipeline.py` : la logique de reponse RAG avec Mistral, le filtrage temporel et le formatage des sources sont implementes.
- `ragas_report.py` : l'evaluation multi-metriques Ragas et l'export de rapports sont pris en charge.
- `data_quality_checks.py` : les controles qualite sur `events_processed.jsonl` sont executes (champs, dates, geographie).
- `__init__.py` : le package Python `API` est declare.

## Lancement rapide

Depuis `rag_runtime`, la sequence suivante est generalement utilisee.

1. Une cle API Mistral est renseignee dans `.env` (requise pour `ask`, `chat`, `report`):

```bash
MISTRAL_API_KEY="votre_cle"
```

2. L'image Docker du RAG est construite:

```bash
docker compose -f docker/docker-compose.yml build rag
```

3. La validite des donnees preprocess est verifiee:

```bash
docker compose -f docker/docker-compose.yml run --rm rag check
```

4. Le mode interactif est demarre pour poser des questions:

```bash
docker compose -f docker/docker-compose.yml run --rm rag chat
```

## Commandes utiles

Question unique (reponse + sources):

```bash
docker compose -f docker/docker-compose.yml run --rm rag ask --question "Propose moi des expositions a Caen"
```

Rebuild index (reconstruction FAISS a partir de `data/data_processed/events_processed.jsonl`):

```bash
docker compose -f docker/docker-compose.yml run --rm rag index --mode build
```

Rapport Ragas (metriques d'evaluation du RAG):

```bash
docker compose -f docker/docker-compose.yml run --rm rag report
```

Aide CLI (liste des commandes disponibles):

```bash
docker compose -f docker/docker-compose.yml run --rm rag --help
```

Arret et suppression de l'instance Docker en cours:

- une interruption du process lance au premier plan est effectuee avec `Ctrl+C`
- un arret explicite du service est possible avec:

```bash
docker compose -f docker/docker-compose.yml stop rag
```

- une suppression du conteneur du service est possible avec:

```bash
docker compose -f docker/docker-compose.yml rm -f rag
```

- un nettoyage complet des conteneurs et du reseau compose est possible avec:

```bash
docker compose -f docker/docker-compose.yml down --remove-orphans
```

Documentation detaillee:

- `docker/README.md`
- `API/README.md`
