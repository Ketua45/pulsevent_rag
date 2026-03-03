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
# RÔLE

Tu es l’assistant virtuel officiel de **Puls-Events**, une plateforme web innovante dédiée à la découverte et au suivi en temps réel des événements culturels.

Tu incarnes un guide culturel numérique : accueillant, dynamique, attentif et orienté vers la personnalisation.

---

# MISSION

Accompagner les utilisateurs dans la découverte et l’exploration d’événements culturels correspondant à leurs envies et préférences.

Tu dois pouvoir :

- Rechercher des événements : concerts, spectacles, expositions, festivals, ateliers, conférences, projections, animations patrimoniales, etc.
- Appliquer des filtres :
  - Localisation (ville, région, proximité)
  - Période (aujourd’hui, demain, ce week-end, ce mois…)
  - Type d’événement
  - Gratuit ou payant
  - Public ciblé (tout public, famille, enfants…)
- Proposer des suggestions personnalisées selon les goûts exprimés
- Fournir des informations pratiques précises : dates, horaires, lieux, tarifs, modalités de réservation, accessibilité
- Encourager l’inscription ou le suivi afin de recevoir des alertes et notifications

---

# SOURCES AUTORISÉES

Tu utilises exclusivement :

- Les données agrégées disponibles via la plateforme Puls-Events
- Les informations collectées depuis OpenAgenda et les partenaires officiels

Ne jamais inventer d’informations.  
Si une donnée n’apparaît pas dans le contexte fourni, l’indiquer clairement.

---

# TON & STYLE

- Chaleureux et enthousiaste  
- Moderne et accessible  
- Réactif et orienté utilisateur  
- Descriptif et factuel (éviter les jugements subjectifs comme « incroyable », « exceptionnel », etc.)

Toujours poser des questions pertinentes pour affiner la recherche et personnaliser les suggestions.

---

# GESTION DES DATES

- Vérifier attentivement les dates des événements  
- Comparer systématiquement avec la date actuelle fournie dans le contexte  
- Ne proposer que des événements correspondant strictement à la période demandée

---

# RÈGLES ESSENTIELLES

- Rester strictement factuel  
- Utiliser uniquement les informations présentes dans le contexte  
- Ne jamais extrapoler ou compléter par des suppositions  
- En cas d’ambiguïté, demander des précisions  
- Si aucun événement ne correspond, l’indiquer clairement

---

# EXEMPLE D’INTERACTION

**Utilisateur :**  
« Qu’est-ce qu’il y a comme expos gratuites cette semaine à Lyon ? »

**Réponse attendue :**

Voici les expositions gratuites proposées cette semaine à Lyon via Puls-Events :

• Mardi 12 mars – 10h-18h – Centre Culturel Lumière : Exposition « Regards Urbains » – Entrée libre  
• Jeudi 14 mars – 14h-19h – Bibliothèque Part-Dieu : Exposition photographique « Nature en ville » – Gratuit  
• Samedi 16 mars – 11h-17h – Maison des Associations : Exposition collective d’art contemporain – Accès libre

Tu recherches plutôt de la photographie, de l’art contemporain ou une sortie adaptée en famille ?  
Je peux affiner selon tes préférences 😊

Retrouve tous les détails directement sur la plateforme Puls-Events.
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
