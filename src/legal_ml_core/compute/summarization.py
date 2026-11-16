"""Text summarisation: extractive (sumy) and abstractive (HuggingFace).

Folded in from `legal-summarizer-service`. The algorithms are unchanged; what
changed is the plumbing: requests are validated by `parse_request` (no
pydantic), jobs run on this package's worker (see worker._summarization_loop)
instead of the service's own Redis queue, and unknown models are rejected
instead of silently falling back to LexRank.

Model ids:
  lex_rank (default), lsa, text_rank, luhn, edmundson   extractive; `sumy:` prefix optional
  hf:<huggingface model id>                             abstractive; only ids the
                                                        deployment allows (see allowed_models)

summary_length: "short" | "medium" | "long", or a positive int (target words).
  extractive:  short/medium/long -> 3/6/10 sentences; int -> ceil(words / 20)
  abstractive: short/medium/long -> (30,130)/(80,250)/(150,400) tokens;
               int -> max = ceil(words * 1.3), min = max // 2

priority.readability (0-1, extractive only): blend weight between the
algorithm's relevance order and closeness to the median sentence length.
priority.relevance (0-1): extractive -- how few extra candidates are drawn
before the readability pass; abstractive -- beam count (2 / 4 / 6).
"""
from __future__ import annotations

import importlib.util
import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

LANGUAGE = "english"
DEFAULT_MODEL = "lex_rank"

EXTRACTIVE_MODELS = [
    {"id": "lex_rank", "label": "LexRank", "description": "Graph-based sentence centrality. Good general purpose."},
    {"id": "lsa", "label": "LSA", "description": "Latent semantic analysis; picks sentences that cover the main topics."},
    {"id": "text_rank", "label": "TextRank", "description": "PageRank over the sentence-similarity graph."},
    {"id": "luhn", "label": "Luhn", "description": "Word frequency and position heuristic. Fastest."},
    {"id": "edmundson", "label": "Edmundson", "description": "Cue words, keywords, title and position; favours sentences that state an obligation."},
]

# The abstractive models a deployment offers unless it configures its own list
# (LEGAL_ML_SUMMARIZATION_HF_MODELS). A request cannot name an arbitrary hub id:
# that would let any caller make the worker download any model.
ABSTRACTIVE_MODELS = [
    {"id": "hf:facebook/bart-large-cnn", "label": "BART large (CNN)", "description": "Strong English abstractive summaries. Slow on CPU."},
    {"id": "hf:sshleifer/distilbart-cnn-12-6", "label": "DistilBART (CNN)", "description": "Lighter, faster BART. English."},
    {"id": "hf:yhavinga/ul2-large-dutch", "label": "UL2 large (Dutch)", "description": "Dutch; suited to Rechtspraak and other NL legal text."},
]

SUMMARY_LENGTHS = ("short", "medium", "long")
MAX_TARGET_WORDS = 5000
DEFAULT_MAX_INPUT_CHARS = 500_000

_EXTRACTIVE_IDS = {m["id"] for m in EXTRACTIVE_MODELS}
_SENTENCES_FOR_LENGTH = {"short": 3, "medium": 6, "long": 10}
_TOKENS_FOR_LENGTH = {"short": (30, 130), "medium": (80, 250), "long": (150, 400)}
_AVG_WORDS_PER_SENTENCE = 20

# Edmundson's cue method needs bonus and stigma words; sumy refuses empty sets
# (legal-summarizer-service passed empty ones, so its edmundson always failed).
# Bonus: deontic markers, which flag the sentences that carry a rule. Stigma:
# words that introduce illustrations rather than the rule itself.
EDMUNDSON_BONUS_WORDS = ("shall", "must", "required", "obliged", "obligation", "prohibited", "entitled", "liable")
EDMUNDSON_STIGMA_WORDS = ("example", "instance", "illustration", "e.g", "hypothetical")
# ~1.3 tokens per word keeps 700 words under BART's 1024-token position limit.
_CHUNK_WORDS = 700

SummaryLength = Union[str, int]
ProgressFn = Callable[[int], None]


@dataclass(frozen=True)
class SummaryRequest:
    input_text: str
    summary_length: SummaryLength = "medium"
    relevance: float = 0.5
    readability: float = 0.5
    model: str = DEFAULT_MODEL

    def to_payload(self) -> dict:
        return {
            "input_text": self.input_text,
            "summary_length": self.summary_length,
            "priority": {"relevance": self.relevance, "readability": self.readability},
            "model": self.model,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "SummaryRequest":
        priority = payload.get("priority") or {}
        return cls(
            input_text=payload["input_text"],
            summary_length=payload.get("summary_length", "medium"),
            relevance=float(priority.get("relevance", 0.5)),
            readability=float(priority.get("readability", 0.5)),
            model=payload.get("model") or DEFAULT_MODEL,
        )


# ---------------------------------------------------------------------------
# Catalogue and validation
# ---------------------------------------------------------------------------

def abstractive_available() -> bool:
    """True when transformers and torch are importable. Checked without
    importing them, so the API process never pays torch's import cost."""
    return all(importlib.util.find_spec(m) is not None for m in ("transformers", "torch"))


def allowed_models(config=None) -> Dict[str, List[dict]]:
    """The catalogue this deployment offers. `LEGAL_ML_SUMMARIZATION_HF_MODELS`
    (a list of `hf:` ids, or of bare hub ids) replaces the default abstractive
    list; an empty list turns abstractive summarisation off."""
    configured = (config or {}).get("LEGAL_ML_SUMMARIZATION_HF_MODELS")
    if configured is None:
        abstractive = list(ABSTRACTIVE_MODELS)
    else:
        known = {m["id"]: m for m in ABSTRACTIVE_MODELS}
        abstractive = []
        for raw in configured:
            model_id = raw if raw.startswith("hf:") else f"hf:{raw}"
            abstractive.append(known.get(model_id) or {"id": model_id, "label": model_id[3:], "description": ""})
    return {"extractive": list(EXTRACTIVE_MODELS), "abstractive": abstractive}


def normalise_model(model: Optional[str], config=None) -> str:
    """Canonical model id, or ValueError naming what is accepted."""
    model = (model or "").strip() or DEFAULT_MODEL
    if model.startswith("sumy:"):
        model = model[len("sumy:"):]
    if model in _EXTRACTIVE_IDS:
        return model
    if model.startswith("hf:"):
        if model in {m["id"] for m in allowed_models(config)["abstractive"]}:
            return model
        raise ValueError(f"Model '{model}' is not offered by this deployment")
    raise ValueError(
        f"Unknown model '{model}'. Use one of {sorted(_EXTRACTIVE_IDS)} or an hf: model from /api/summarize/models"
    )


def _unit_interval(value, name) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"priority.{name} must be a number between 0 and 1")
    return float(value)


def parse_request(body, config=None) -> SummaryRequest:
    """Validate a POST /api/summarize body. Raises ValueError with a message
    fit to return to the caller."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")

    text = body.get("input_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input_text is required")
    max_chars = (config or {}).get("LEGAL_ML_SUMMARIZATION_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS)
    if len(text) > max_chars:
        raise ValueError(f"input_text is longer than {max_chars} characters")

    length = body.get("summary_length", "medium")
    if isinstance(length, bool) or not (
        length in SUMMARY_LENGTHS or (isinstance(length, int) and 0 < length <= MAX_TARGET_WORDS)
    ):
        raise ValueError(f"summary_length must be one of {list(SUMMARY_LENGTHS)} or a word count from 1 to {MAX_TARGET_WORDS}")

    priority = body.get("priority")
    if priority is None:
        priority = {}
    if not isinstance(priority, dict):
        raise ValueError("priority must be an object")
    relevance = _unit_interval(priority.get("relevance", 0.5), "relevance")
    readability = _unit_interval(priority.get("readability", 0.5), "readability")

    return SummaryRequest(
        input_text=text,
        summary_length=length,
        relevance=relevance,
        readability=readability,
        model=normalise_model(body.get("model"), config),
    )


# ---------------------------------------------------------------------------
# Running a summary
# ---------------------------------------------------------------------------

def summarize(request: SummaryRequest, on_progress: Optional[ProgressFn] = None) -> str:
    """Route to the abstractive or extractive backend. `on_progress(pct)` is
    called with 0-100 as work advances (per chunk for abstractive)."""
    if request.model.startswith("hf:"):
        return _summarize_hf(request, on_progress)
    return _summarize_sumy(request)


NLTK_RESOURCES = (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"))


def ensure_nltk_data() -> None:
    """sumy's English tokenizer needs NLTK's punkt data. Download it when it
    is missing; failure is logged, and surfaces as a job error later."""
    import nltk

    for name, path in NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as e:  # network down in a sealed deployment
                logger.warning("could not download NLTK %s: %s", name, e)


def _target_sentence_count(summary_length) -> int:
    if isinstance(summary_length, int):
        return max(1, math.ceil(summary_length / _AVG_WORDS_PER_SENTENCE))
    return _SENTENCES_FOR_LENGTH.get(summary_length, 6)


def _build_sumy_summarizer(algorithm: str):
    from sumy.nlp.stemmers import Stemmer
    from sumy.summarizers.edmundson import EdmundsonSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.utils import get_stop_words

    classes = {
        "lex_rank": LexRankSummarizer,
        "lsa": LsaSummarizer,
        "text_rank": TextRankSummarizer,
        "luhn": LuhnSummarizer,
        "edmundson": EdmundsonSummarizer,
    }
    cls = classes[algorithm]
    summarizer = cls(Stemmer(LANGUAGE))
    if cls is EdmundsonSummarizer:
        summarizer.null_words = get_stop_words(LANGUAGE)
        summarizer.bonus_words = EDMUNDSON_BONUS_WORDS
        summarizer.stigma_words = EDMUNDSON_STIGMA_WORDS
    else:
        summarizer.stop_words = get_stop_words(LANGUAGE)
    return summarizer


def readability_pass(sentences: List, target: int, readability: float) -> List:
    """Re-rank candidates (given in relevance order) by
    `(1 - readability) * relevance_rank + readability * length_penalty`,
    both normalised to [0, 1], and return the best `target` in document order.
    At readability 0 this is the algorithm's own top `target`."""
    n = len(sentences)
    if readability <= 0 or n <= target:
        return sentences[:target]

    word_counts = [len(str(s).split()) for s in sentences]
    median_len = sorted(word_counts)[n // 2]
    max_dev = max(abs(c - median_len) for c in word_counts) or 1

    def score(i: int) -> float:
        relevance_rank = i / (n - 1) if n > 1 else 0.0
        length_penalty = abs(word_counts[i] - median_len) / max_dev
        return (1 - readability) * relevance_rank + readability * length_penalty

    selected = sorted(sorted(range(n), key=score)[:target])
    return [sentences[i] for i in selected]


def _summarize_sumy(request: SummaryRequest) -> str:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser

    target = _target_sentence_count(request.summary_length)
    # High relevance trusts the algorithm's ranking: draw no extra candidates.
    # Extras only matter when the readability pass is engaged.
    extra = 0 if request.relevance >= 0.7 else math.ceil((1 - request.relevance) * 4)
    wanted = target + extra if request.readability > 0 else target

    parser = PlaintextParser.from_string(request.input_text, Tokenizer(LANGUAGE))
    doc_sentences = len(parser.document.sentences)
    if doc_sentences == 0:
        return request.input_text.strip()

    summarizer = _build_sumy_summarizer(request.model)
    candidates = list(summarizer(parser.document, min(wanted, doc_sentences)))
    return " ".join(str(s) for s in readability_pass(candidates, target, request.readability))


# Loaded pipelines, by hub id, for the life of the worker process.
_PIPELINE_CACHE: Dict[str, object] = {}


def _load_pipeline(model_id: str):
    if model_id not in _PIPELINE_CACHE:
        if not abstractive_available():
            raise RuntimeError("Abstractive summarisation needs transformers and torch, which this deployment does not have")
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info("loading summarisation model %s (device %s)", model_id, device)
        _PIPELINE_CACHE[model_id] = pipeline("summarization", model=model_id, device=device, truncation=True)
    return _PIPELINE_CACHE[model_id]


def _token_range(summary_length) -> tuple:
    if isinstance(summary_length, int):
        tokens = max(30, math.ceil(summary_length * 1.3))
        return max(20, tokens // 2), tokens
    return _TOKENS_FOR_LENGTH.get(summary_length, _TOKENS_FOR_LENGTH["medium"])


def _num_beams(relevance: float) -> int:
    if relevance >= 0.8:
        return 6
    if relevance >= 0.5:
        return 4
    return 2


def chunk_text(text: str, max_words: int = _CHUNK_WORDS) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _summarize_hf(request: SummaryRequest, on_progress: Optional[ProgressFn]) -> str:
    pipe = _load_pipeline(request.model[len("hf:"):])
    min_len, max_len = _token_range(request.summary_length)
    chunks = chunk_text(request.input_text)
    # Split the budget across chunks so the whole stays near the requested length.
    chunk_max = max(30, max_len // len(chunks))
    chunk_min = max(10, min_len // len(chunks))
    beams = _num_beams(request.relevance)

    parts = []
    for i, chunk in enumerate(chunks):
        result = pipe(chunk, max_length=chunk_max, min_length=chunk_min, num_beams=beams, do_sample=False, truncation=True)
        text = result[0]["summary_text"].strip()
        if text:
            parts.append(text)
        if on_progress:
            on_progress(10 + int(85 * (i + 1) / len(chunks)))
    return " ".join(parts)
