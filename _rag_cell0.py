import os
import re
import math
from collections import Counter, defaultdict

import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOF_PATTERNS = [
    "Abstract Factory", "Adapter", "Bridge", "Builder", "Command",
    "Composite", "Decorator", "Facade", "Factory Method", "Flyweight",
    "Interpreter", "Iterator", "Mediator", "Memento", "Observer",
    "Prototype", "Proxy", "Singleton", "State", "Strategy",
    "Template Method", "Visitor", "Null Object",
]

PATTERN_ALIASES = {
    re.sub(r"[^a-z]", "", p.lower()): p for p in GOF_PATTERNS
}
PATTERN_ALIASES.update({
    "factorymethod": "Factory Method",
    "abstractfactory": "Abstract Factory",
    "templatemethod": "Template Method",
    "nullobject": "Null Object",
})

LABEL_COMMENT_RE = re.compile(
    r"^\s*'.*(?:design\s*pattern|dp\s*role|gof)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
CLASS_RE = re.compile(
    r"^\s*(abstract\s+)?(class|interface|enum)\s+\"?([\w.]+)\"?",
    re.IGNORECASE,
)
METHOD_RE = re.compile(
    r"^\s*(?:[+\-#~]\s*)?(?:\{(?:static|abstract|classifier)\}\s*)*"
    r"(?:<<\w+>>\s*)?([A-Za-z_][\w]*)\s*\(",
)
REL_RE = re.compile(
    r"([\w.]+)(?:\s+\"[^\"]*\")?\s*(<\|--|<\|\.\.|\*--|o--|-->|<--|\.\.>)\s*([\w.]+)"
)

QUERY_PREFIX = (
    "Represent this UML class structure for retrieving similar "
    "software design-pattern diagrams: "
)

CUE_RULES = [
    ("singleton", ("getinstance", "instance", "private_constructor", "static_instance")),
    ("observer", ("notify", "update", "addlistener", "removelistener", "addobserver",
                  "removeobserver", "propertychange", "firepropertychange", "subscribe")),
    ("strategy", ("setstrategy", "strategy", "algorithm")),
    ("state", ("setstate", "getstate", "changestate", "handle")),
    ("command", ("execute", "undo", "redo", "invoke", "actionperformed")),
    ("factory_method", ("create", "factorymethod", "newinstance", "make")),
    ("abstract_factory", ("createproduct", "abstractfactory")),
    ("builder", ("build", "director", "construct")),
    ("prototype", ("clone", "copy", "prototype")),
    ("iterator", ("hasnext", "next", "iterator", "hasmoreelements")),
    ("composite", ("addchild", "removechild", "getchild", "children", "addcomponent")),
    ("decorator", ("decorator", "wrap", "component")),
    ("adapter", ("adapt", "wrapper", "adaptee")),
    ("proxy", ("proxy", "realsubject")),
    ("template_method", ("template", "hook", "primitiveoperation")),
    ("visitor", ("accept", "visit")),
    ("memento", ("savestate", "restorestate", "getmemento", "setmemento", "creatememento")),
    ("facade", ("facade",)),
]


def normalize_pattern_name(name):
    if not name:
        return "Unknown"
    name = os.path.splitext(str(name).strip())[0]
    name = re.split(r"\s*-\s*\d", name, maxsplit=1)[0].strip()
    key = re.sub(r"[^a-z]", "", name.lower())
    return PATTERN_ALIASES.get(key, name.strip())


def simple_name(qualified):
    return qualified.split(".")[-1]


def camel_tokens(name):
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"[_\\-]+", " ", parts)
    return [p.lower() for p in parts.split() if p]


def strip_gold_labels(text):
    """Remove P-MART gold comments so retrieval cannot cheat on labels."""
    text = LABEL_COMMENT_RE.sub("", text)
    text = re.sub(r"^\s*'+\s*$", "", text, flags=re.MULTILINE)
    return text


def extract_cues(method_names, fields, constructors):
    tokens = []
    for m in method_names:
        tokens.extend(camel_tokens(m))
        tokens.append(re.sub(r"[^a-z]", "", m.lower()))
    field_blob = " ".join(f.lower() for f in fields)
    method_blob = " ".join(tokens)
    ctor_blob = " ".join(c.lower() for c in constructors)

    cues = []
    if any(
        c.strip().startswith("-") or "private" in c.lower()
        for c in constructors
    ):
        cues.append("private_constructor")
    if re.search(r"\{static\}.*instance|static_instance|instance\s*:", field_blob, re.I):
        cues.append("static_instance")
    if "getinstance" in method_blob:
        cues.append("getinstance")

    for cue_name, keys in CUE_RULES:
        if any(k in method_blob or k in field_blob or k in ctor_blob for k in keys):
            cues.append(cue_name)
    return cues


def parse_puml_structure(text):
    types = []
    methods_by_type = defaultdict(list)
    fields_by_type = defaultdict(list)
    constructors = []
    relations = []
    current = None
    brace_depth = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("'") or line.startswith("@") or line.startswith("package"):
            continue

        class_match = CLASS_RE.match(line)
        if class_match:
            abstract, kind, name = class_match.groups()
            kind = kind.lower()
            if abstract or "abstract class" in line.lower():
                kind = "abstract_class"
            current = simple_name(name)
            types.append((kind, current))
            continue

        if current and "{" in line:
            brace_depth += line.count("{")
        if current and "}" in line:
            brace_depth -= line.count("}")
            if brace_depth <= 0:
                current = None
                brace_depth = 0

        rel = REL_RE.search(line)
        if rel:
            left, op, right = rel.groups()
            relations.append((simple_name(left), op, simple_name(right)))
            continue

        if current:
            method = METHOD_RE.search(raw_line)
            if method:
                mname = method.group(1)
                methods_by_type[current].append(mname)
                if ("<<Create>>" in line) or (mname.lower() == current.lower()):
                    constructors.append(line)
                continue
            if ":" in line and "(" not in line:
                field = line.split(":")[0]
                field = re.sub(r"[{}]", " ", field)
                fields_by_type[current].append(field.strip())

    return {
        "types": types,
        "methods": methods_by_type,
        "fields": fields_by_type,
        "constructors": constructors,
        "relations": relations,
    }


def describe_relations(relations):
    op_map = {
        "<|--": "inherits",
        "<|..": "implements",
        "*--": "composes",
        "o--": "aggregates",
        "-->": "associates",
        "<--": "associated_from",
        "..>": "depends",
    }
    lines = []
    for left, op, right in relations:
        if left.startswith(("java", "Java", "javax")) or right.startswith(("java", "Java")):
            continue
        verb = op_map.get(op, "related")
        # PlantUML: Target <|-- Child  => Child inherits Target
        if op in ("<|--", "<|.."):
            lines.append(f"{right} {verb} {left}")
        elif op == "<--":
            lines.append(f"{right} associates {left}")
        else:
            lines.append(f"{left} {verb} {right}")
    return lines[:40]


def semantic_puml_processor(text, include_roles=False):
    """
    Build a label-free structural description for embedding.
    Gold DP-role / pattern-name comments are ignored so KB and test diagrams
    live in the same representation space.
    """
    cleaned = strip_gold_labels(text)
    parsed = parse_puml_structure(cleaned)

    type_bits = [f"{kind} {name}" for kind, name in parsed["types"][:30]]
    method_names = [m for ms in parsed["methods"].values() for m in ms]
    field_names = [f for fs in parsed["fields"].values() for f in fs]
    method_bits = []
    for tname, methods in list(parsed["methods"].items())[:25]:
        uniq = sorted(set(methods))[:12]
        if uniq:
            method_bits.append(f"{tname} methods " + " ".join(uniq))

    rel_bits = describe_relations(parsed["relations"])
    cues = extract_cues(method_names, field_names, parsed["constructors"])

    tokens = []
    if include_roles:
        roles = re.findall(r"DP role:\s*(\w+)", text, re.IGNORECASE)
        tokens.extend(f"role_{r.lower()}" for r in roles)

    tokens.extend(type_bits)
    tokens.extend(method_bits)
    tokens.extend(rel_bits)
    if cues:
        tokens.append("cues " + " ".join(sorted(set(cues))))

    # Keep a small amount of raw method vocabulary for lexical match.
    tokens.extend(camel_tokens(" ".join(method_names[:80])))
    return " ".join(tokens)


def compact_summary(text, max_chars=1800):
    cleaned = strip_gold_labels(text)
    parsed = parse_puml_structure(cleaned)
    lines = ["Participants:"]
    for kind, name in parsed["types"][:20]:
        methods = ", ".join(sorted(set(parsed["methods"].get(name, [])))[:8])
        extra = f" [{methods}]" if methods else ""
        lines.append(f"- {kind} {name}{extra}")
    rels = describe_relations(parsed["relations"])
    if rels:
        lines.append("Relations:")
        lines.extend(f"- {r}" for r in rels[:25])
    method_names = [m for ms in parsed["methods"].values() for m in ms]
    field_names = [f for fs in parsed["fields"].values() for f in fs]
    cues = extract_cues(method_names, field_names, parsed["constructors"])
    if cues:
        lines.append("Structural cues: " + ", ".join(sorted(set(cues))))
    summary = "\n".join(lines)
    return summary[:max_chars]


def load_and_process_docs(folder_path, exclude_ids=None):
    exclude_ids = set(exclude_ids or [])
    documents = []
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(folder_path)

    for file in sorted(os.listdir(folder_path)):
        if file in exclude_ids or not file.endswith((".puml", ".plantuml")):
            continue
        path = os.path.join(folder_path, file)
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            raw_content = handle.read()
        semantic_content = semantic_puml_processor(raw_content)
        if not semantic_content.strip():
            continue
        documents.append({
            "id": file,
            "label": normalize_pattern_name(file),
            "content": semantic_content,
            "summary": compact_summary(raw_content),
            "raw": raw_content,
        })
    return documents


class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-base-en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)

    def encode(self, texts, batch_size=16, is_query=False, show_progress_bar=True):
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )


class VectorStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []

    def add_embeddings(self, embeddings, metadatas):
        self.index.add(np.asarray(embeddings, dtype="float32"))
        self.metadata.extend(metadatas)

    def search(self, query_embedding, top_k=5):
        query_embedding = np.asarray(query_embedding, dtype="float32")
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            res = self.metadata[idx].copy()
            res["dense_score"] = float(distances[0][i])
            results.append(res)
        return results


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.tokenized = [self._tokenize(d["content"]) for d in documents]
        self.doc_len = [len(toks) or 1 for toks in self.tokenized]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df = Counter()
        for toks in self.tokenized:
            df.update(set(toks))
        n = len(self.tokenized) or 1
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9_]+", text.lower())

    def search(self, query_text, top_k=10):
        q_tokens = self._tokenize(query_text)
        scores = []
        for i, toks in enumerate(self.tokenized):
            tf = Counter(toks)
            dl = self.doc_len[i]
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        results = []
        for score, i in scores[:top_k]:
            res = self.docs[i].copy()
            res["bm25_score"] = float(score)
            results.append(res)
        return results


def reciprocal_rank_fusion(result_lists, k=60, top_k=5):
    fused = defaultdict(lambda: {"score": 0.0, "doc": None})
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            entry = fused[doc["id"]]
            entry["score"] += 1.0 / (k + rank)
            entry["doc"] = doc
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    out = []
    for item in ranked[:top_k]:
        doc = item["doc"].copy()
        doc["score"] = item["score"]
        out.append(doc)
    return out


def vote_patterns(retrieved_docs):
    votes = defaultdict(float)
    for doc in retrieved_docs:
        votes[doc["label"]] += doc.get("score", doc.get("dense_score", 1.0))
    if not votes:
        return "Unknown", {}
    best = max(votes, key=votes.get)
    return best, dict(votes)


def llm_reasoning(query_raw, retrieved_docs, vote_label, vote_scores):
    examples = []
    for doc in retrieved_docs:
        examples.append(
            f"Example pattern: {doc['label']}\n"
            f"Structural summary:\n{doc.get('summary') or compact_summary(doc['raw'])}"
        )
    vote_txt = ", ".join(
        f"{k}={v:.3f}" for k, v in sorted(vote_scores.items(), key=lambda kv: -kv[1])
    ) or "none"
    allowed = ", ".join(sorted({d["label"] for d in retrieved_docs} | set(GOF_PATTERNS)))
    prompt = f"""
You are a software architecture expert specializing in Gang of Four design patterns.

Task: identify the GoF design pattern implemented by the QUERY diagram.
Use the retrieved examples as analogical evidence. Prefer structural roles,
relationships, and method cues over class names. The query has no gold labels.

Retrieved similarity votes: {vote_txt}
Vote guess: {vote_label}

QUERY STRUCTURE:
{compact_summary(query_raw, max_chars=2500)}

RETRIEVED EXAMPLES:
{chr(10).join(examples)}

Rules:
- Answer with exactly one pattern name from: {allowed}
- Typical confusions: State vs Strategy vs Template Method; Adapter vs Proxy vs Decorator; Factory Method vs Abstract Factory.
- If evidence is mixed, pick the pattern whose retrieved examples best match the query relations and cues.
"""
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": "You detect GoF design patterns from UML structure. Reply with only the pattern name."},
            {"role": "user", "content": prompt},
        ],
    )
    return normalize_pattern_name(response.choices[0].message.content.strip())


def build_rag_index(folder_path, exclude_folder=None):
    exclude_ids = set()
    if exclude_folder and os.path.isdir(exclude_folder):
        exclude_ids = {f for f in os.listdir(exclude_folder) if f.endswith((".puml", ".plantuml"))}
        print(f"Excluding {len(exclude_ids)} filenames that also appear in the test folder.")

    print("Loading PlantUML files...")
    docs = load_and_process_docs(folder_path, exclude_ids=exclude_ids)
    print(f"Indexed {len(docs)} diagrams.")

    texts = [doc["content"] for doc in docs]
    embedder = EmbeddingModel()
    embeddings = embedder.encode(texts, is_query=False)

    vector_store = VectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add_embeddings(embeddings, docs)
    bm25 = BM25Index(docs)
    return vector_store, embedder, bm25


def detect_pattern(test_doc, vector_store, embedder, bm25=None, top_k=5, use_llm=True):
    query_text = semantic_puml_processor(test_doc["content"])
    query_embedding = embedder.encode(
        [query_text], is_query=True, show_progress_bar=False
    )
    dense_k = max(top_k * 4, 12)
    dense_hits = vector_store.search(query_embedding, top_k=dense_k)

    if bm25 is not None:
        sparse_hits = bm25.search(query_text, top_k=dense_k)
        results = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k)
    else:
        results = dense_hits[:top_k]
        for r in results:
            r["score"] = r.get("dense_score", 0.0)

    vote_label, vote_scores = vote_patterns(results)
    retrieved_ids = [r["id"] for r in results]
    retrieved_labels = [r["label"] for r in results]

    if use_llm:
        llm_prediction = llm_reasoning(
            test_doc["content"], results, vote_label, vote_scores
        )
        return llm_prediction, retrieved_ids, retrieved_labels, vote_label

    return vote_label, retrieved_ids, retrieved_labels, vote_label


def evaluate_detection(test_folder, vector_store, embedder, bm25=None, top_k=5, use_llm=True):
    test_docs = load_and_process_docs(test_folder)
    correct_counts = Counter()
    total_counts = Counter()
    retrieval_hits = 0
    vote_correct = 0

    for doc in tqdm(test_docs):
        true_label = normalize_pattern_name(doc["id"])
        total_counts[true_label] += 1

        predicted_label, retrieved_ids, retrieved_labels, vote_label = detect_pattern(
            {"content": doc["raw"]},
            vector_store,
            embedder,
            bm25=bm25,
            top_k=top_k,
            use_llm=use_llm,
        )

        if true_label in retrieved_labels:
            retrieval_hits += 1
        if vote_label == true_label:
            vote_correct += 1
        if predicted_label == true_label:
            correct_counts[true_label] += 1

        print(
            f"{doc['id']} | True: {true_label} | Pred: {predicted_label} "
            f"| Vote: {vote_label} | Hit@{top_k}: {true_label in retrieved_labels} "
            f"| Retrieved: {retrieved_labels}"
        )

    per_pattern_accuracy = {
        label: correct_counts[label] / total_counts[label]
        for label in total_counts
    }
    n = sum(total_counts.values()) or 1
    overall_accuracy = sum(correct_counts.values()) / n
    retrieval_at_k = retrieval_hits / n
    vote_accuracy = vote_correct / n

    print("\nPer-Pattern Accuracy:")
    for label, acc in sorted(per_pattern_accuracy.items()):
        print(f"{label}: {acc:.4f} ({correct_counts[label]}/{total_counts[label]})")

    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"Retrieval Hit@{top_k}: {retrieval_at_k:.4f}")
    print(f"Hybrid-vote Accuracy: {vote_accuracy:.4f}")
    return overall_accuracy, per_pattern_accuracy
