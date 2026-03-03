import argparse
import os
from datetime import datetime, timezone

from mistralai import Mistral

try:
    from .paths import PROCESSED_JSONL_PATH, VECTORSTORE_DIR
    from .vectorstore_index import load_documents_from_jsonl, load_vectorstore
except ImportError:
    from paths import PROCESSED_JSONL_PATH, VECTORSTORE_DIR
    from vectorstore_index import load_documents_from_jsonl, load_vectorstore


SYSTEM_TEMPLATE = """
### ROLE
### RÔLE :
Vous êtes l'assistant virtuel officiel de Puls-Events, la plateforme web innovante dédiée à la découverte et au suivi en temps réel d'événements culturels.  
Agissez comme un guide culturel numérique accueillant, enthousiaste, réactif et personnalisé.

### OBJECTIF :
Aider les utilisateurs à découvrir, explorer et suivre des événements culturels adaptés à leurs préférences :  
- Rechercher des concerts, spectacles, expositions, festivals, ateliers, conférences, animations patrimoniales, etc.  
- Filtrer par lieu (ville, région, proximité), période (aujourd'hui, ce week-end, ce mois…), type d'événement, gratuit/payant, pour tous publics/enfants…  
- Proposer des suggestions personnalisées en fonction des goûts exprimés  
- Fournir des infos pratiques : dates, horaires, lieux, tarifs, réservation, accessibilité  
- Encourager l'inscription ou le suivi pour recevoir des alertes/notifications

### SOURCES AUTORISÉES :
- Données agrégées via la plateforme Puls-Events (collectées depuis OpenAgenda et sources partenaires officielles)
- Utilise UNIQUEMENT les informations présentes dans le contexte fourni
- Si l'information n'est pas dans le contexte, indique-le clairement

### COMPORTEMENT & STYLE :
Ton : Chaleureux, enthousiaste, convivial, moderne et accessible  
Précision : Informations exactes, à jour et uniquement tirées des données de la plateforme  
Personnalisation : Poser des questions pour affiner les suggestions
Enthousiasme : Valoriser la richesse culturelle sans exagération
Ambiguïté : Demander poliment des précisions si nécessaire
Info Manquante : Si l'événement n'est pas dans le contexte, l'indiquer clairement

### RÈGLES IMPORTANTES :
- Rester factuel : utiliser uniquement les informations du contexte fourni
- Être descriptif plutôt que subjectif (éviter "incontournable", "génial")
- DATES : Comparer attentivement les dates des événements avec la date actuelle fournie en début de contexte

### EXEMPLE D'INTERACTION GUIDÉE :
Utilisateur : « Quels événements sympas ce week-end à Paris ? »
Assistant Attendu :
« Voici ce que Puls-Events a en stock pour ce week-end à Paris :
• Vendredi 14 janv. 20h – Théâtre de l'Athénée : Concert jazz « Vibes d'hiver » – Tarif 18 € / 12 € réduit
• Samedi 15 janv. 14h-18h – Musée Guimet : Atelier famille « Manga, tout un art! » – Gratuit sur inscription
• Dimanche 16 janv. 11h – Comédie-Française : Représentation théâtrale – Entrée à partir de 25€
Tu préfères quelque chose de gratuit, en intérieur, ou plutôt musical ? Je peux affiner selon tes goûts ! 😊
Retrouve tous les détails et réserve sur la plateforme Puls-Events. »
"""


def parse_iso_date(value: str):
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) >= 5 and text[-5] in ["+", "-"] and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_upcoming(doc, reference_time):
    dt = parse_iso_date(doc.metadata.get("date_start", ""))
    return dt is not None and dt >= reference_time


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


class PulsEventRAG:
    def __init__(self, retriever, event_catalog_by_id, event_catalog_by_slug, api_key: str | None = None):
        self.retriever = retriever
        self.event_catalog_by_id = event_catalog_by_id
        self.event_catalog_by_slug = event_catalog_by_slug

        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is missing.")
        self.client = Mistral(api_key=key)

    def _event_record_from_doc(self, doc):
        event_id = safe_text(doc.metadata.get("event_id"))
        slug = safe_text(doc.metadata.get("slug"))
        if event_id and event_id in self.event_catalog_by_id:
            return self.event_catalog_by_id[event_id]
        if slug and slug in self.event_catalog_by_slug:
            return self.event_catalog_by_slug[slug]
        return {}

    def _meta_or_record(self, doc, field: str):
        record = self._event_record_from_doc(doc)
        return safe_text(record.get(field) or doc.metadata.get(field, ""))

    def _format_location(self, doc):
        location_name = self._meta_or_record(doc, "location_name")
        address = self._meta_or_record(doc, "address")
        postal_code = self._meta_or_record(doc, "postal_code")
        city = self._meta_or_record(doc, "city")
        department = self._meta_or_record(doc, "department")

        line = ", ".join([p for p in [location_name, address] if p])
        city_block = " ".join([p for p in [postal_code, city] if p]).strip()
        if city_block:
            line = ", ".join([p for p in [line, city_block] if p])
        if department:
            line = f"{line} ({department})" if line else department
        return line or "Lieu non renseigne"

    def _build_event_context(self, doc):
        title = self._meta_or_record(doc, "title") or "Titre non renseigne"
        slug = self._meta_or_record(doc, "slug") or "slug-non-renseigne"
        date_start = self._meta_or_record(doc, "date_start")
        date_end = self._meta_or_record(doc, "date_end")
        if date_start and date_end:
            date_text = f"{date_start} -> {date_end}"
        else:
            date_text = date_start or date_end or "Date non renseignee"

        excerpt = safe_text(doc.page_content)
        if len(excerpt) > 900:
            excerpt = excerpt[:900] + "..."

        return "\n".join(
            [
                "EVENT",
                f"Titre: {title}",
                f"Lieu: {self._format_location(doc)}",
                f"Date: {date_text}",
                f"Slug: {slug}",
                f"Extrait: {excerpt}",
            ]
        )

    def answer(self, question: str, k: int = 6, model: str = "mistral-small-latest", fetch_k: int = 60):
        reference_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        candidate_docs = self.retriever.vectorstore.similarity_search(question, k=max(k, fetch_k))

        docs = []
        seen = set()
        for doc in candidate_docs:
            event_key = doc.metadata.get("event_id") or doc.metadata.get("slug") or doc.page_content
            if event_key in seen:
                continue
            if is_upcoming(doc, reference_time):
                docs.append(doc)
                seen.add(event_key)
            if len(docs) >= k:
                break

        if not docs:
            return "Je ne trouve pas d'evenement a venir correspondant dans les donnees disponibles.", []

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
        system = f"{SYSTEM_TEMPLATE}\nDate courante: {now}"
        context = "\n\n".join([self._build_event_context(doc) for doc in docs])
        prompt = f"{system}\n\nCONTEXTE:\n{context}\n\nQUESTION: {question}\nREPONSE:"

        response = self.client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, docs


def build_rag_pipeline(
    jsonl_path=PROCESSED_JSONL_PATH,
    vectorstore_dir=VECTORSTORE_DIR,
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    search_k: int = 10,
    api_key: str | None = None,
) -> PulsEventRAG:
    _, event_catalog_by_id, event_catalog_by_slug = load_documents_from_jsonl(jsonl_path)
    vectorstore, _ = load_vectorstore(vectorstore_dir=vectorstore_dir, embeddings_model=embeddings_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    return PulsEventRAG(
        retriever=retriever,
        event_catalog_by_id=event_catalog_by_id,
        event_catalog_by_slug=event_catalog_by_slug,
        api_key=api_key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PulsEvent RAG from disk assets.")
    parser.add_argument("--question", default="Presente toi et propose moi des expositions")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    rag = build_rag_pipeline()
    answer, sources = rag.answer(args.question, k=args.k)
    print("Reponse:\n")
    print(answer)
    print("\nSources:")
    for doc in sources[:5]:
        print("-", doc.metadata.get("title"), "|", doc.metadata.get("city"), "|", doc.metadata.get("slug"))


if __name__ == "__main__":
    main()
