# Docker RAG PulsEvent

La configuration Docker du RAG est centralisee dans ce dossier.

## Prerequis

Les elements suivants sont requis:

- Docker et Docker Compose
- variable `MISTRAL_API_KEY` renseignee dans `rag_runtime/.env` (requise pour `ask` et `report`)

Le vectorstore est charge depuis `../vectorstore_normandie` (bind mount local du runtime).
Le cache Hugging Face est charge depuis `../.cache/huggingface` (bind mount local du runtime).

## Commandes rapides

Depuis la racine `rag_runtime`, les commandes suivantes peuvent etre executees.

Build de l'image:

```bash
docker compose -f docker/docker-compose.yml build rag
```

Aide CLI:

```bash
docker compose -f docker/docker-compose.yml run --rm rag --help
```

Controle qualite:

```bash
docker compose -f docker/docker-compose.yml run --rm rag check
```

Question unique:

```bash
docker compose -f docker/docker-compose.yml run --rm rag ask --question "Propose moi des expositions a Caen"
```

## CLI interactif dans Docker

Une session interactive peut etre ouverte avec:

```bash
docker compose -f docker/docker-compose.yml run --rm rag chat
```

La sortie du chat est realisee avec `/quit`.

Un shell Docker peut aussi etre ouvert, puis le CLI peut etre lance:

```bash
docker compose -f docker/docker-compose.yml run --rm --entrypoint sh rag
python -m API.main chat
```

Si le vectorstore est vide, une reconstruction peut etre lancee avec:

```bash
docker compose -f docker/docker-compose.yml run --rm rag index --mode build
```

## Arret et suppression de l'instance en cours

Une interruption du process actif au premier plan est realisee avec `Ctrl+C`.

Un arret explicite du service peut etre realise avec:

```bash
docker compose -f docker/docker-compose.yml stop rag
```

Une suppression du conteneur du service peut etre realisee avec:

```bash
docker compose -f docker/docker-compose.yml rm -f rag
```

Un nettoyage complet des conteneurs et du reseau compose peut etre realise avec:

```bash
docker compose -f docker/docker-compose.yml down --remove-orphans
```

## Depannage cache Hugging Face

Si un telechargement est interrompu, un lock peut rester present.
Les locks locaux peuvent etre supprimes avec:

```bash
find .cache/huggingface -name "*.lock" -delete
```
