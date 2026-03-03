import argparse
import importlib
import math
import re
import types
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import pandas as pd

try:
    from .paths import PROCESSED_DIR
    from .rag_pipeline import build_rag_pipeline
except ImportError:
    from paths import PROCESSED_DIR
    from rag_pipeline import build_rag_pipeline


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _is_metric_instance(obj):
    return (
        obj is not None
        and not isinstance(obj, types.ModuleType)
        and not isinstance(obj, type)
        and hasattr(obj, "required_columns")
        and hasattr(obj, "name")
    )


def _metric_factory(metric_obj):
    if _is_metric_instance(metric_obj):
        return metric_obj
    if isinstance(metric_obj, type):
        for kwargs in ({}, {"strictness": 1}):
            try:
                cand = metric_obj(**kwargs)
                if _is_metric_instance(cand):
                    return cand
            except Exception:
                pass
        return None
    if callable(metric_obj):
        for kwargs in ({}, {"strictness": 1}):
            try:
                cand = metric_obj(**kwargs)
                if _is_metric_instance(cand):
                    return cand
            except Exception:
                pass
        return None
    return None


def _load_metric(attr_candidates):
    for module_name in ("ragas.metrics", "ragas.metrics.collections"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                mod = importlib.import_module(module_name)
        except Exception:
            continue

        for attr in attr_candidates:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    if not hasattr(mod, attr):
                        continue
                    raw = getattr(mod, attr)
            except Exception:
                continue

            if isinstance(raw, types.ModuleType):
                continue
            metric = _metric_factory(raw)
            if _is_metric_instance(metric):
                return metric
    return None


def _import_ragas_components():
    from ragas import evaluate

    llm_wrapper_cls = None
    emb_wrapper_cls = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from ragas.llms import LangchainLLMWrapper as _LangchainLLMWrapper
        llm_wrapper_cls = _LangchainLLMWrapper
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from ragas.embeddings import LangchainEmbeddingsWrapper as _LangchainEmbeddingsWrapper
        emb_wrapper_cls = _LangchainEmbeddingsWrapper
    except Exception:
        pass
    return evaluate, llm_wrapper_cls, emb_wrapper_cls


def _build_dataset(samples):
    from datasets import Dataset

    cols = {k: [s.get(k) for s in samples] for k in samples[0].keys()}
    return Dataset.from_dict(cols)


class _HashEmbeddings:
    """Simple fallback embeddings if vectorstore embeddings are not accessible."""

    def __init__(self, dim=256):
        self.dim = dim

    def _embed(self, text):
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", str(text).lower()):
            vec[hash(token) % self.dim] += 1.0
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _tokenize(text):
    return {m.group(0).lower() for m in TOKEN_RE.finditer(str(text))}


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b)) or 1
    return inter / union


def _lexical_relevancy(question, response):
    q = _tokenize(question)
    r = _tokenize(response)
    if not q:
        return 0.0
    return len(q.intersection(r)) / len(q)


def _collect_numeric_values(value):
    out = []
    if value is None or isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        vf = float(value)
        if math.isfinite(vf):
            out.append(vf)
        return out
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return out
        try:
            vf = float(txt)
            if math.isfinite(vf):
                out.append(vf)
            return out
        except Exception:
            return out
    if isinstance(value, dict):
        for key in ("score", "value", "mean", "avg"):
            if key in value:
                out.extend(_collect_numeric_values(value[key]))
        if out:
            return out
        for v in value.values():
            out.extend(_collect_numeric_values(v))
        return out
    if isinstance(value, (list, tuple, set)):
        for v in value:
            out.extend(_collect_numeric_values(v))
        return out
    for attr in ("score", "value", "mean", "avg"):
        if hasattr(value, attr):
            out.extend(_collect_numeric_values(getattr(value, attr)))
            if out:
                return out
    if hasattr(value, "__dict__"):
        out.extend(_collect_numeric_values(vars(value)))
        if out:
            return out
    try:
        vf = float(value)
        if math.isfinite(vf):
            out.append(vf)
    except Exception:
        pass
    return out


def _extract_metric_values(result, aliases):
    aliases = tuple(a.lower() for a in aliases)

    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            for col in df.columns:
                if any(a in str(col).lower() for a in aliases):
                    vals = []
                    for x in df[col].tolist():
                        vals.extend(_collect_numeric_values(x))
                    if vals:
                        return vals
        except Exception:
            pass

    if hasattr(result, "scores"):
        try:
            rows = result.scores
            if isinstance(rows, list):
                values = []
                for row in rows:
                    items = row.items() if isinstance(row, dict) else getattr(row, "__dict__", {}).items()
                    for key, val in items:
                        if any(a in str(key).lower() for a in aliases):
                            values.extend(_collect_numeric_values(val))
                if values:
                    return values
        except Exception:
            pass

    if hasattr(result, "to_dict"):
        try:
            data = result.to_dict()
            if isinstance(data, dict):
                values = []
                for key, val in data.items():
                    if any(a in str(key).lower() for a in aliases):
                        values.extend(_collect_numeric_values(val))
                if values:
                    return values
        except Exception:
            pass
    return []


def _flatten_required_columns(metric):
    req = getattr(metric, "required_columns", None)
    if req is None:
        return set()
    if isinstance(req, dict):
        cols = set()
        for val in req.values():
            if isinstance(val, set):
                cols.update(val)
            elif isinstance(val, (list, tuple)):
                cols.update(set(val))
        return cols
    if isinstance(req, set):
        return req
    return set()


def _evaluate_with_fallback_wrappers(evaluate, dataset, metrics, base_llm, base_embeddings, llm_wrapper_cls, emb_wrapper_cls):
    try:
        return evaluate(dataset=dataset, metrics=metrics, llm=base_llm, embeddings=base_embeddings), "native"
    except Exception as native_exc:
        llm_obj = base_llm
        emb_obj = base_embeddings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if llm_wrapper_cls is not None:
                try:
                    llm_obj = llm_wrapper_cls(base_llm)
                except Exception:
                    llm_obj = base_llm
            if emb_wrapper_cls is not None:
                try:
                    emb_obj = emb_wrapper_cls(base_embeddings)
                except Exception:
                    emb_obj = base_embeddings
        try:
            return evaluate(dataset=dataset, metrics=metrics, llm=llm_obj, embeddings=emb_obj), "wrapped"
        except Exception as wrapped_exc:
            raise RuntimeError(
                f"Evaluation Ragas failed (native: {native_exc}; wrapped: {wrapped_exc})"
            ) from wrapped_exc


def _build_reference_from_docs(docs, max_items=3):
    lines = []
    for doc in docs[:max_items]:
        md = getattr(doc, "metadata", {}) or {}
        title = str(md.get("title") or "").strip() or "Titre inconnu"
        city = str(md.get("city") or "").strip() or "Ville inconnue"
        date_start = str(md.get("date_start") or "").strip() or "Date inconnue"
        slug = str(md.get("slug") or "").strip() or "slug inconnu"
        lines.append(f"- {title} | {city} | {date_start} | {slug}")
    if lines:
        return "\n".join(lines)
    return "Aucun evenement pertinent trouve."


def _safe_mean(values):
    return mean(values) if values else float("nan")


def run_report(questions: list[str] | None = None):
    if questions is None:
        questions = [
            "Propose moi des evenements a Caen",
            "Je cherche des expositions en Normandie",
            "Y a-t-il des concerts a Rouen ?",
            "Quels evenements famille ce week-end en Normandie ?",
            "Je veux des activites culturelles gratuites",
        ]

    rag = build_rag_pipeline()

    samples = []
    sample_custom_rows = []
    for question in questions:
        answer, docs = rag.answer(question)
        response_text = str(answer)
        contexts = [d.page_content for d in docs]
        if not contexts:
            continue

        reference_text = _build_reference_from_docs(docs)
        samples.append(
            {
                "user_input": question,
                "response": response_text,
                "retrieved_contexts": contexts,
                "reference": reference_text,
                "question": question,
                "answer": response_text,
                "contexts": contexts,
                "ground_truth": reference_text,
            }
        )

        q_toks = _tokenize(question)
        r_toks = _tokenize(response_text)
        c_toks = _tokenize("\n".join(contexts))
        sample_custom_rows.append(
            {
                "question": question,
                "num_contexts": len(contexts),
                "response_chars": len(response_text),
                "contexts_chars": sum(len(c) for c in contexts),
                "q_response_overlap": _lexical_relevancy(question, response_text),
                "q_context_jaccard": _jaccard(q_toks, c_toks),
                "response_context_jaccard": _jaccard(r_toks, c_toks),
            }
        )

    if not samples:
        raise RuntimeError("No context returned by rag.answer. Report cannot be built.")

    metric_specs = [
        {
            "id": "faithfulness",
            "description": "La reponse est-elle soutenue par le contexte recupere ?",
            "attrs": ["faithfulness", "Faithfulness"],
            "aliases": ["faithfulness"],
        },
        {
            "id": "answer_relevancy",
            "description": "La reponse est-elle pertinente pour la question ?",
            "attrs": ["ResponseRelevancy", "response_relevancy", "answer_relevancy", "AnswerRelevancy"],
            "aliases": ["answer_relevancy", "response_relevancy", "relevancy"],
        },
        {
            "id": "context_precision",
            "description": "Les contextes recuperes sont-ils vraiment utiles ?",
            "attrs": ["context_precision", "ContextPrecision", "LLMContextPrecisionWithoutReference"],
            "aliases": ["context_precision", "llm_context_precision_without_reference"],
        },
        {
            "id": "context_recall",
            "description": "Les contextes recuperes couvrent-ils les infos necessaires ?",
            "attrs": ["context_recall", "ContextRecall", "LLMContextRecall"],
            "aliases": ["context_recall"],
        },
        {
            "id": "answer_similarity",
            "description": "Similarite entre reponse et reference.",
            "attrs": ["answer_similarity", "AnswerSimilarity", "semantic_similarity", "SemanticSimilarity"],
            "aliases": ["answer_similarity", "semantic_similarity"],
        },
        {
            "id": "answer_correctness",
            "description": "Exactitude globale de la reponse.",
            "attrs": ["answer_correctness", "AnswerCorrectness"],
            "aliases": ["answer_correctness"],
        },
        {
            "id": "noise_sensitivity",
            "description": "Sensibilite de la reponse au bruit dans le contexte.",
            "attrs": ["noise_sensitivity", "NoiseSensitivity"],
            "aliases": ["noise_sensitivity"],
        },
    ]

    evaluate, llm_wrapper_cls, emb_wrapper_cls = _import_ragas_components()
    dataset = _build_dataset(samples)
    dataset_columns = set(samples[0].keys())

    # Ragas evaluation expects a LangChain-compatible LLM.
    try:
        from langchain_mistralai.chat_models import ChatMistralAI
    except Exception:
        from langchain_mistralai import ChatMistralAI

    eval_model = os.environ.get("RAGAS_EVALUATOR_MODEL", "mistral-small-latest")
    base_llm = ChatMistralAI(
        model=eval_model,
        temperature=0,
        api_key=os.environ.get("MISTRAL_API_KEY"),
    )

    # Reuse vectorstore embeddings when possible.
    base_embeddings = getattr(rag.retriever, "vectorstore", None)
    if base_embeddings is not None and hasattr(base_embeddings, "embedding_function"):
        base_embeddings = base_embeddings.embedding_function
    else:
        base_embeddings = _HashEmbeddings()

    report_rows = []

    for spec in metric_specs:
        metric_obj = _load_metric(spec["attrs"])
        if metric_obj is None:
            report_rows.append(
                {
                    "metric": spec["id"],
                    "description": spec["description"],
                    "category": "ragas",
                    "status": "not_available",
                    "mode": "n/a",
                    "n": 0,
                    "mean": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "std": float("nan"),
                    "error": "metric not available in current ragas version",
                }
            )
            continue

        missing_cols = sorted(list(_flatten_required_columns(metric_obj) - dataset_columns))
        if missing_cols:
            report_rows.append(
                {
                    "metric": spec["id"],
                    "description": spec["description"],
                    "category": "ragas",
                    "status": "skipped_missing_columns",
                    "mode": "n/a",
                    "n": 0,
                    "mean": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "std": float("nan"),
                    "error": f"missing columns: {missing_cols}",
                }
            )
            continue

        try:
            result, mode = _evaluate_with_fallback_wrappers(
                evaluate,
                dataset,
                [metric_obj],
                base_llm,
                base_embeddings,
                llm_wrapper_cls,
                emb_wrapper_cls,
            )
            values = _extract_metric_values(result, tuple(spec["aliases"] + [getattr(metric_obj, "name", spec["id"])]))
            if not values:
                report_rows.append(
                    {
                        "metric": spec["id"],
                        "description": spec["description"],
                        "category": "ragas",
                        "status": "no_numeric_values",
                        "mode": mode,
                        "n": 0,
                        "mean": float("nan"),
                        "min": float("nan"),
                        "max": float("nan"),
                        "std": float("nan"),
                        "error": "no numeric values",
                    }
                )
            else:
                report_rows.append(
                    {
                        "metric": spec["id"],
                        "description": spec["description"],
                        "category": "ragas",
                        "status": "ok",
                        "mode": mode,
                        "n": len(values),
                        "mean": _safe_mean(values),
                        "min": min(values),
                        "max": max(values),
                        "std": pd.Series(values).std(ddof=0),
                        "error": "",
                    }
                )
        except Exception as exc:
            report_rows.append(
                {
                    "metric": spec["id"],
                    "description": spec["description"],
                    "category": "ragas",
                    "status": "error",
                    "mode": "n/a",
                    "n": 0,
                    "mean": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "std": float("nan"),
                    "error": str(exc),
                }
            )

    sample_custom_df = pd.DataFrame(sample_custom_rows)
    custom_metric_descriptions = {
        "num_contexts": "Nombre de contexts recuperes.",
        "response_chars": "Longueur de la reponse en caracteres.",
        "contexts_chars": "Longueur totale des contexts en caracteres.",
        "q_response_overlap": "Overlap lexical question/reponse.",
        "q_context_jaccard": "Jaccard lexical question/contexts.",
        "response_context_jaccard": "Jaccard lexical reponse/contexts.",
    }
    for col in [
        "num_contexts",
        "response_chars",
        "contexts_chars",
        "q_response_overlap",
        "q_context_jaccard",
        "response_context_jaccard",
    ]:
        vals = [float(x) for x in sample_custom_df[col].tolist()]
        report_rows.append(
            {
                "metric": col,
                "description": custom_metric_descriptions[col],
                "category": "custom",
                "status": "ok",
                "mode": "computed",
                "n": len(vals),
                "mean": _safe_mean(vals),
                "min": min(vals),
                "max": max(vals),
                "std": pd.Series(vals).std(ddof=0),
                "error": "",
            }
        )

    report_df = pd.DataFrame(report_rows).sort_values(by=["category", "metric"]).reset_index(drop=True)
    return report_df, sample_custom_df


def save_report(report_df: pd.DataFrame, sample_df: pd.DataFrame) -> dict[str, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_csv = PROCESSED_DIR / f"ragas_report_{ts}.csv"
    report_json = PROCESSED_DIR / f"ragas_report_{ts}.json"
    sample_csv = PROCESSED_DIR / f"ragas_sample_report_{ts}.csv"

    report_df.to_csv(report_csv, index=False)
    report_df.to_json(report_json, orient="records", force_ascii=False, indent=2)
    sample_df.to_csv(sample_csv, index=False)
    return {
        "report_csv": report_csv,
        "report_json": report_json,
        "sample_csv": sample_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-metric Ragas report from existing RAG.")
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="Optional txt file with one question per line.",
    )
    args = parser.parse_args()

    questions = None
    if args.questions_file:
        questions = [line.strip() for line in args.questions_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    report_df, sample_df = run_report(questions=questions)
    outputs = save_report(report_df, sample_df)

    print("Ragas report generated:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")
    print("\nSummary:")
    print(report_df[["metric", "description", "category", "status", "mean", "mode"]])


if __name__ == "__main__":
    main()
