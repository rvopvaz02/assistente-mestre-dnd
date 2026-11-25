"""Busca híbrida: BM25 (léxico) + denso (embeddings no ChromaDB), fusão por RRF e reordenação opcional.

Por que híbrida: perguntas de RPG misturam termos exatos ("meia-cobertura", "+2", "D20"), em que o
BM25 é forte, com paráfrases ("ficar atrás de uma mureta"), em que o vetor denso é forte.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property

from mestre import config
from mestre.rag.ingest import Chunk

STOP = set("""a o e é de da do das dos em no na nos nas um uma uns umas para por com sem que se ao aos à às
ou como mais mas seu sua seus suas ele ela eles elas isso isto este esta qual quais quando onde quanto
quantos quantas ser ter tem há pode podem posso faz fazer sobre entre até já não sim me te lhe eu você""".split())

PERMISSOES = {"jogadores": {"jogadores"}, "mestre": {"jogadores", "mestre"}}


def tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    toks = re.findall(r"[a-z0-9+]+", t)
    return [w for w in toks if w not in STOP and len(w) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float            # score final usado para ordenar
    bm25_rank: int | None = None
    dense_rank: int | None = None
    dense_sim: float | None = None
    rerank: float | None = None


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embed_model: str | None = None,
                 rerank_model: str | None = None, persist: bool = True):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.embed_model_name = embed_model or config.EMBED_MODEL
        self.rerank_model_name = rerank_model or config.RERANK_MODEL
        self.persist = persist
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(self._doc_text(c)) for c in chunks])
        self._collection = self._build_dense()

    @staticmethod
    def _doc_text(c: Chunk) -> str:
        return f"{c.secao}. {c.texto}"

    # ---------- denso ----------
    @cached_property
    def _embedder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.embed_model_name)

    def _encode(self, texts: list[str], is_query: bool):
        if "e5" in self.embed_model_name.lower():
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        return self._embedder.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def _build_dense(self):
        import chromadb

        if self.persist:
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        else:
            client = chromadb.EphemeralClient()
        name = "dnd_" + re.sub(r"[^a-zA-Z0-9]", "_", self.embed_model_name.split("/")[-1])[:40]
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        have = set(col.get(include=[])["ids"]) if col.count() else set()
        want = {c.id for c in self.chunks}
        stale = list(have - want)
        if stale:
            col.delete(ids=stale)
        new = [c for c in self.chunks if c.id not in have]
        for i in range(0, len(new), 128):
            batch = new[i:i + 128]
            col.add(ids=[c.id for c in batch], embeddings=self._encode([self._doc_text(c) for c in batch], False),
                    metadatas=[{"fonte": c.fonte, "permissao": c.permissao, "versao": c.versao} for c in batch])
        return col

    # ---------- reranker ----------
    @cached_property
    def _reranker(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.rerank_model_name, max_length=512)

    # ---------- busca ----------
    def search(self, query: str, k: int = 5, mode: str = "hybrid", rerank: bool = True,
               perfil: str = "jogadores", pool: int = 30) -> list[Hit]:
        allowed = PERMISSOES.get(perfil, {"jogadores"})
        ok = [i for i, c in enumerate(self.chunks) if c.permissao in allowed]
        ok_ids = {self.chunks[i].id for i in ok}

        bm25_order: list[int] = []
        if mode in ("bm25", "hybrid"):
            scores = self._bm25.get_scores(tokenize(query))
            bm25_order = sorted(ok, key=lambda i: -scores[i])[:pool]

        dense_order: list[str] = []
        sims: dict[str, float] = {}
        if mode in ("dense", "hybrid"):
            res = self._collection.query(query_embeddings=self._encode([query], True),
                                         n_results=min(pool * 2, len(self.chunks)),
                                         where={"permissao": {"$in": sorted(allowed)}})
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                if cid in ok_ids:
                    dense_order.append(cid)
                    sims[cid] = 1 - dist
            dense_order = dense_order[:pool]

        # Fusão RRF (k=60): robusta a escalas diferentes de score.
        fused: dict[str, Hit] = {}
        for r, i in enumerate(bm25_order):
            c = self.chunks[i]
            fused.setdefault(c.id, Hit(c, 0.0)).bm25_rank = r + 1
        for r, cid in enumerate(dense_order):
            h = fused.setdefault(cid, Hit(self.by_id[cid], 0.0))
            h.dense_rank, h.dense_sim = r + 1, sims[cid]
        for h in fused.values():
            h.score = sum(1 / (60 + x) for x in (h.bm25_rank, h.dense_rank) if x)
        hits = sorted(fused.values(), key=lambda h: -h.score)[:pool]

        if rerank and hits:
            scores = self._reranker.predict([(query, self._doc_text(h.chunk)) for h in hits])
            for h, s in zip(hits, scores):
                h.rerank = float(s)
                h.score = float(s)
            hits.sort(key=lambda h: -h.score)
        return hits[:k]
