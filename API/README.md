# API PulsEvent

Une version modulaire du notebook `PulsEvent_POC.ipynb` est fournie dans ce dossier.
Une execution reproductible du pipeline RAG peut ainsi etre realisee sans notebook.

## Vue d'ensemble

Le package est organise par responsabilite:

- `paths.py`: les chemins projet et artefacts sont centralises.
- `fetch_openagenda.py`: la collecte des donnees brutes OpenAgenda est realisee.
- `preprocess_events.py`: le nettoyage et la transformation vers les formats RAG sont appliques.
- `vectorstore_index.py`: la creation et l'interrogation de l'index FAISS sont prises en charge.
- `rag_pipeline.py`: la logique de reponse basee sur le retriever est implementee.
- `ragas_report.py`: l'evaluation multi-metriques avec Ragas et l'export des rapports sont geres.
- `data_quality_checks.py`: les controles qualite du fichier `events_processed.jsonl` sont executes.
- `main.py`: un point d'entree unique de pilotage est expose.

## Prerequis

Les elements suivants sont requis:

- Python 3.11 (recommande dans l'environnement `PulsEnv`)
- dependances projet installees (pandas, langchain, mistralai, ragas, etc.)
- variable d'environnement `MISTRAL_API_KEY` definie pour `ask` et `report`

Exemple:

```bash
export MISTRAL_API_KEY="votre_cle_api"
```

## Point d'entree principal

Depuis la racine du projet, les deux formes suivantes peuvent etre utilisees:

```bash
python API/main.py --help
python -m API.main --help
```

## Utilisation avec Docker

La conteneurisation est supportee par:

- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `.dockerignore` a la racine (utilise pour le contexte de build)

Configuration actuelle:

- image CPU-only (aucune dependance CUDA/NVIDIA n'est installee dans le conteneur)
- vectorstore charge depuis `../vectorstore_normandie` (bind mount local du dossier runtime)
- cache Hugging Face charge depuis `../.cache/huggingface` (bind mount local)

### Build de l'image

La commande suivante est utilisee pour construire l'image:

```bash
docker compose -f docker/docker-compose.yml build rag
```

### Variables d'environnement

Pour la generation (`ask`) et l'evaluation (`report`), `MISTRAL_API_KEY` est requise.

```bash
export MISTRAL_API_KEY="votre_cle_api"
```

### Aide generale

La liste des sous-commandes est accessible avec:

```bash
docker compose -f docker/docker-compose.yml run --rm rag --help
```

### Exemples de commandes Docker

Pipeline complet (collecte + preprocessing + index):

```bash
docker compose -f docker/docker-compose.yml run --rm rag pipeline --fetch-mode export
```

Question unique au RAG:

```bash
docker compose -f docker/docker-compose.yml run --rm rag ask --question "Propose moi des expositions a Caen"
```

CLI interactif RAG:

```bash
docker compose -f docker/docker-compose.yml run --rm rag chat
```

Rapport Ragas:

```bash
docker compose -f docker/docker-compose.yml run --rm rag report
```

Controles qualite:

```bash
docker compose -f docker/docker-compose.yml run --rm rag check
```

### Volumes montes par Docker Compose

- `../data -> /app/data` (bind mount)
- `../vectorstore_normandie -> /app/vectorstore_normandie` (bind mount)
- `../.cache/huggingface -> /cache/huggingface` (bind mount)

La persistance des artefacts est ainsi assuree (donnees preprocess, index FAISS, cache modeles).

## Commandes disponibles

### 1) Collecte des donnees brutes

La collecte via export CSV est realisee avec:

```bash
python API/main.py fetch --mode export --region Normandie
```

Une collecte paginee peut etre executee avec:

```bash
python API/main.py fetch --mode paginated --region Normandie --limit 100 --sleep 0.2
```

### 2) Preprocessing des donnees

Le preprocessing complet est lance avec:

```bash
python API/main.py preprocess --stage all
```

Stages disponibles:

- `window`: `evenements_normandie.csv` -> `normandie_1y_data.json`
- `processed`: `normandie_1y_data.json` -> `events_processed.jsonl`
- `all`: les deux etapes sont executees

### 3) Indexation FAISS

Construction:

```bash
python API/main.py index --mode build
```

Recherche rapide:

```bash
python API/main.py index --mode search --query "concert jazz a Rouen" --k 5
```

### 4) Question au pipeline RAG

```bash
python API/main.py ask --question "Propose moi des expositions a Caen"
```

Parametres utiles:

- `--k`: nombre final d'evenements renvoyes
- `--fetch-k`: taille du pool de retrieval avant filtrage temporel
- `--model`: modele Mistral de generation

### 5) Rapport d'evaluation Ragas

```bash
python API/main.py report
```

Un fichier de questions personnalise peut etre fourni avec:

```bash
python API/main.py report --questions-file data/data_processed/questions_eval.txt
```

Sorties produites:

- `data/data_processed/ragas_report_<timestamp>.csv`
- `data/data_processed/ragas_report_<timestamp>.json`
- `data/data_processed/ragas_sample_report_<timestamp>.csv`

### 6) Controles qualite des donnees

```bash
python API/main.py check
```

Un chemin personnalise peut etre defini avec:

```bash
python API/main.py check --path data/data_processed/events_processed.jsonl
```

### 7) Pipeline complet (collecte -> preprocessing -> index)

```bash
python API/main.py pipeline --fetch-mode export
```

## Flux recommande en production locale

L'enchainement suivant est recommande:

1. `fetch`
2. `preprocess --stage all`
3. `index --mode build`
4. `ask` pour les tests fonctionnels
5. `report` pour l'evaluation qualite
6. `check` pour la validation des donnees

## Notes d'exploitation

- les scripts peuvent etre utilises en mode module (`python -m API.main ...`) ou script (`python API/main.py ...`)
- un cout API peut etre observe pour les commandes de reporting et de generation selon le volume de questions
- les fichiers de sortie sont ecrits dans `data/data_processed` et peuvent etre archives
