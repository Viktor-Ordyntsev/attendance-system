# recognition-client/app/core/matcher.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CandidateMatch:
    participant_id: int
    label: str
    score: float

@dataclass
class MatchResult:
    participant_id: int | None
    label: str
    score: float
    is_match: bool
    top_candidates: list[CandidateMatch]
    reason: str


class FaceMatcher:
    def __init__(
        self,
        threshold: float = 0.55,
        min_margin: float = 0.02,
    ) -> None:
        """
        threshold:
            Минимальный cosine similarity для принятия совпадения.

        min_margin:
            Минимальный отрыв лучшего кандидата от второго.
            Нужен, чтобы отбрасывать неоднозначные совпадения.
        """
        self.threshold = threshold
        self.min_margin = min_margin
        self.reference_db: list[dict] = []

    def set_reference_db(self, reference_db: list[dict]) -> None:
        """
        Загружает эталонную базу embeddings.
        """
        prepared_db: list[dict] = []

        for person in reference_db:
            embedding = np.asarray(person["embedding"], dtype=np.float32)

            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue

            embedding = embedding / norm

            if embedding.ndim != 1:
                raise ValueError(
                    f"Embedding for participant {person['participant_id']} must be 1D"
                )

            prepared_db.append(
                {
                    "participant_id": int(person["participant_id"]),
                    "label": str(person["label"]),
                    "embedding": embedding,
                }
            )

        self.reference_db = prepared_db

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Считает косинусное сходство между двумя векторами.

        Формула:
            sim(a, b) = (a · b) / (||a|| * ||b||)

        Чем ближе к 1.0, тем вектора более похожи.
        """
        return float(np.dot(a, b)) 
    def _build_candidates(self, query_embedding: np.ndarray) -> list[CandidateMatch]:
        """
        Для query embedding строит список кандидатов:
        каждому участнику соответствует score сходства.
        """
        candidates: list[CandidateMatch] = []

        for person in self.reference_db:
            score = self.cosine_similarity(query_embedding, person["embedding"])
            candidates.append(
                CandidateMatch(
                    participant_id=person["participant_id"],
                    label=person["label"],
                    score=score,
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def match(self, embedding: np.ndarray, top_k: int = 3) -> MatchResult:
        """
        Основной метод сопоставления.

        Шаги:
        1. Проверяем входной embedding
        2. Строим список кандидатов
        3. Берем best и second-best
        4. Применяем threshold
        5. Применяем margin
        6. Возвращаем MatchResult
        """
        query = np.asarray(embedding, dtype=np.float32)
        
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all():
            return MatchResult(
                participant_id=None,
                label="unknown",
                score=0.0,
                is_match=False,
                top_candidates=[],
                reason="non_finite_query_embedding",
            )

        query = query / norm

        if query.ndim != 1:
            raise ValueError("Query embedding must be 1D")

        if not self.reference_db:
            return MatchResult(
                participant_id=None,
                label="unknown",
                score=0.0,
                is_match=False,
                top_candidates=[],
                reason="empty_reference_db",
            )

        candidates = self._build_candidates(query)
        top_candidates = candidates[:top_k]

        best = top_candidates[0]
        second = top_candidates[1] if len(top_candidates) > 1 else None

        # Проверка 1: лучший кандидат должен пройти threshold
        if best.score < self.threshold:
            return MatchResult(
                participant_id=None,
                label="unknown",
                score=best.score,
                is_match=False,
                top_candidates=top_candidates,
                reason="below_threshold",
            )

        # Проверка 2: лучший кандидат должен достаточно оторваться от второго
        if second is not None:
            margin = best.score - second.score
            if margin < self.min_margin:
                return MatchResult(
                    participant_id=None,
                    label="unknown",
                    score=best.score,
                    is_match=False,
                    top_candidates=top_candidates,
                    reason="low_margin_to_second_candidate",
                )

        return MatchResult(
            participant_id=best.participant_id,
            label=best.label,
            score=best.score,
            is_match=True,
            top_candidates=top_candidates,
            reason="matched",
        )