"""
Item Canonicalization Service - Main pipeline orchestrator.

Combines normalization, barcode lookup, fuzzy matching, and embedding search
to map free text/OCR/barcode to canonical items in items_catalog.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..utils.text_normalizer import TextNormalizer
from ..models.fuzzy_matcher import FuzzyMatcher
from ..models.embedding_matcher import EmbeddingMatcher
from ..config import (
    ItemMatchResult,
    ConfidenceLevel,
    HUMAN_REVIEW_CONFIDENCE_THRESHOLD,
)
from ...shared.models_v2 import ItemsCatalog


logger = logging.getLogger(__name__)


@dataclass
class CanonicalizationPipeline:
    """Complete item canonicalization pipeline."""
    
    text_normalizer: TextNormalizer
    fuzzy_matcher: FuzzyMatcher
    embedding_matcher: EmbeddingMatcher
    confidence_threshold: float = HUMAN_REVIEW_CONFIDENCE_THRESHOLD
    
    def __post_init__(self):
        """Initialize pipeline components."""
        logger.info("Initialized Item Canonicalization Pipeline")


class ItemCanonicalizationService:
    """Service for mapping user input to canonical items."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_matcher: Optional[EmbeddingMatcher] = None,
    ):
        """
        Initialize canonicalization service.
        
        Args:
            db_session: Database session for catalog lookups
            embedding_matcher: Pre-initialized embedding matcher (optional)
        """
        self.db = db_session
        
        # Initialize components
        self.text_normalizer = TextNormalizer()
        self.fuzzy_matcher = FuzzyMatcher()
        
        # Embedding matcher (lazy initialization)
        self._embedding_matcher = embedding_matcher
        
        logger.info("ItemCanonicalizationService initialized")
    
    async def canonicalize(
        self,
        user_input: str,
        barcode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ItemMatchResult:
        """
        Main canonicalization pipeline.
        
        Pipeline stages:
        1. Normalization: Clean and standardize input text
        2. Barcode lookup: Direct match if barcode provided
        3. Exact match: Check for exact text matches
        4. Fuzzy match: Token sort ratio matching
        5. Embedding match: Semantic similarity search
        6. Scoring: Combine scores and rank candidates
        7. Confidence check: Route to human review if needed
        
        Args:
            user_input: Raw text (OCR, manual entry, etc.)
            barcode: Optional barcode string
            metadata: Additional context (brand, category hints, etc.)
            
        Returns:
            ItemMatchResult with best match and confidence
        """
        start_time = datetime.now()
        metadata = metadata or {}
        
        logger.info(f"Canonicalizing: '{user_input}' (barcode: {barcode})")
        
        # Stage 1: Normalization
        normalized = self.text_normalizer.normalize(user_input)
        logger.debug(f"Normalized: '{normalized.normalized}'")
        
        # Stage 2: Barcode lookup (highest confidence)
        if barcode:
            barcode_result = await self._barcode_lookup(barcode)
            if barcode_result:
                logger.info(f"Barcode match found: {barcode_result.canonical_item_id}")
                return barcode_result
        
        # Stage 3: Exact match check
        exact_result = await self._exact_match(normalized.normalized)
        if exact_result:
            logger.info(f"Exact match found: {exact_result.canonical_item_id}")
            return exact_result
        
        # Stage 4 & 5: Fuzzy + Embedding matches (run in parallel)
        fuzzy_candidates = await self._fuzzy_match(normalized.normalized)
        embedding_candidates = await self._embedding_match(normalized.normalized)
        
        # Stage 6: Combine and score candidates
        final_result = self._combine_scores(
            user_input=user_input,
            normalized=normalized,
            fuzzy_candidates=fuzzy_candidates,
            embedding_candidates=embedding_candidates,
            metadata=metadata,
        )
        
        # Stage 7: Human review check
        if final_result.match_confidence < self.fuzzy_matcher.threshold / 100:
            final_result.needs_human_review = True
            logger.info(
                f"Low confidence ({final_result.match_confidence:.2f}), "
                "routing to human review"
            )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Canonicalization complete: {final_result.canonical_item_id} "
            f"(confidence: {final_result.match_confidence:.2f}, "
            f"elapsed: {elapsed:.3f}s)"
        )
        
        return final_result
    
    async def _barcode_lookup(self, barcode: str) -> Optional[ItemMatchResult]:
        """Direct database lookup by barcode."""
        query = select(ItemsCatalog).where(ItemsCatalog.barcode == barcode)
        result = await self.db.execute(query)
        item = result.scalar_one_or_none()
        
        if item:
            return ItemMatchResult(
                canonical_item_id=str(item.item_id),
                match_confidence=1.0,  # Perfect match
                match_method="barcode",
                candidate_items=[{
                    "item_id": str(item.item_id),
                    "name": item.name,
                    "category": item.category,
                    "brand": item.brand,
                }],
                needs_human_review=False,
                explanation={
                    "method": "barcode_lookup",
                    "barcode": barcode,
                },
            )
        
        return None
    
    async def _exact_match(self, normalized_text: str) -> Optional[ItemMatchResult]:
        """Check for exact normalized name match."""
        # Query database for exact match (case-insensitive)
        query = select(ItemsCatalog).where(
            ItemsCatalog.name_normalized == normalized_text
        )
        result = await self.db.execute(query)
        item = result.scalar_one_or_none()
        
        if item:
            return ItemMatchResult(
                canonical_item_id=str(item.item_id),
                match_confidence=0.95,  # Very high confidence
                match_method="exact",
                candidate_items=[{
                    "item_id": str(item.item_id),
                    "name": item.name,
                    "category": item.category,
                    "brand": item.brand,
                }],
                needs_human_review=False,
                explanation={
                    "method": "exact_match",
                    "normalized_query": normalized_text,
                },
            )
        
        return None
    
    async def _fuzzy_match(self, normalized_text: str) -> List[Dict[str, Any]]:
        """Fuzzy string matching against catalog."""
        # Fetch candidate items from database
        # (In production, cache this or use materialized view)
        query = select(ItemsCatalog).limit(10000)  # Limit for performance
        result = await self.db.execute(query)
        items = result.scalars().all()
        
        if not items:
            return []
        
        # Extract texts and IDs
        candidate_texts = [item.name_normalized or item.name for item in items]
        candidate_ids = [str(item.item_id) for item in items]
        
        # Fuzzy match
        fuzzy_results = self.fuzzy_matcher.match_best(
            normalized_text,
            candidate_texts,
            candidate_ids,
            limit=5,
            use_ensemble=True,
        )
        
        # Convert to dict format
        candidates = []
        for fr in fuzzy_results:
            # Find original item
            item = next(
                (i for i in items if str(i.item_id) == fr.matched_item_id),
                None,
            )
            if item:
                candidates.append({
                    "item_id": fr.matched_item_id,
                    "name": item.name,
                    "category": item.category,
                    "brand": item.brand,
                    "fuzzy_score": fr.score,
                    "match_type": fr.match_type,
                })
        
        return candidates
    
    async def _embedding_match(
        self,
        normalized_text: str,
    ) -> List[Dict[str, Any]]:
        """Semantic similarity matching using embeddings."""
        if self._embedding_matcher is None:
            logger.warning("Embedding matcher not initialized, skipping")
            return []
        
        try:
            # Search FAISS index
            embedding_results = self._embedding_matcher.search(
                normalized_text,
                top_k=5,
                min_similarity=0.5,
            )
            
            # Fetch full item details from database
            item_ids = [er.matched_item_id for er in embedding_results]
            query = select(ItemsCatalog).where(
                ItemsCatalog.item_id.in_(item_ids)
            )
            result = await self.db.execute(query)
            items = {str(item.item_id): item for item in result.scalars().all()}
            
            # Convert to dict format
            candidates = []
            for er in embedding_results:
                item = items.get(er.matched_item_id)
                if item:
                    candidates.append({
                        "item_id": er.matched_item_id,
                        "name": item.name,
                        "category": item.category,
                        "brand": item.brand,
                        "embedding_similarity": er.cosine_similarity,
                    })
            
            return candidates
        
        except Exception as e:
            logger.error(f"Embedding match failed: {e}")
            return []
    
    def _combine_scores(
        self,
        user_input: str,
        normalized: Any,
        fuzzy_candidates: List[Dict[str, Any]],
        embedding_candidates: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> ItemMatchResult:
        """
        Combine fuzzy and embedding scores using logistic combiner.
        
        Final score = α * fuzzy_score + β * embedding_similarity + γ * barcode_bonus
        """
        # Weights for ensemble scoring
        FUZZY_WEIGHT = 0.6
        EMBEDDING_WEIGHT = 0.4
        BARCODE_BONUS = 0.1  # Bonus if barcode present
        
        # Merge candidates by item_id
        candidates_map: Dict[str, Dict[str, Any]] = {}
        
        # Add fuzzy candidates
        for fc in fuzzy_candidates:
            item_id = fc["item_id"]
            candidates_map[item_id] = {
                **fc,
                "fuzzy_score": fc.get("fuzzy_score", 0.0),
                "embedding_similarity": 0.0,
            }
        
        # Add/merge embedding candidates
        for ec in embedding_candidates:
            item_id = ec["item_id"]
            if item_id in candidates_map:
                candidates_map[item_id]["embedding_similarity"] = ec["embedding_similarity"]
            else:
                candidates_map[item_id] = {
                    **ec,
                    "fuzzy_score": 0.0,
                    "embedding_similarity": ec["embedding_similarity"],
                }
        
        # Calculate ensemble scores
        for item_id, candidate in candidates_map.items():
            fuzzy_score = candidate["fuzzy_score"]
            embedding_score = candidate["embedding_similarity"]
            barcode_bonus = BARCODE_BONUS if candidate.get("barcode") else 0.0
            
            ensemble_score = (
                FUZZY_WEIGHT * fuzzy_score +
                EMBEDDING_WEIGHT * embedding_score +
                barcode_bonus
            )
            
            # Apply normalization penalty
            ensemble_score += normalized.confidence_adjustment
            
            # Clip to [0, 1]
            ensemble_score = max(0.0, min(1.0, ensemble_score))
            
            candidate["ensemble_score"] = ensemble_score
        
        # Sort by ensemble score
        sorted_candidates = sorted(
            candidates_map.values(),
            key=lambda x: x["ensemble_score"],
            reverse=True,
        )
        
        # Best match
        if sorted_candidates:
            best_match = sorted_candidates[0]
            
            return ItemMatchResult(
                canonical_item_id=best_match["item_id"],
                match_confidence=best_match["ensemble_score"],
                match_method="ensemble",
                candidate_items=sorted_candidates[:5],  # Top 5
                needs_human_review=best_match["ensemble_score"] < HUMAN_REVIEW_CONFIDENCE_THRESHOLD,
                explanation={
                    "fuzzy_score": best_match["fuzzy_score"],
                    "embedding_similarity": best_match["embedding_similarity"],
                    "normalization_penalty": normalized.confidence_adjustment,
                    "weights": {
                        "fuzzy": FUZZY_WEIGHT,
                        "embedding": EMBEDDING_WEIGHT,
                    },
                },
            )
        else:
            # No match found
            return ItemMatchResult(
                canonical_item_id=None,
                match_confidence=0.0,
                match_method="none",
                candidate_items=[],
                needs_human_review=True,
                explanation={
                    "reason": "no_candidates_found",
                    "user_input": user_input,
                },
            )


# ============================================================================
# Human-in-the-Loop Correction Storage
# ============================================================================

async def store_human_correction(
    db: AsyncSession,
    user_input: str,
    barcode: Optional[str],
    corrected_item_id: str,
    user_id: str,
) -> None:
    """
    Store human correction for future training.
    
    This creates a training sample to improve the model.
    """
    # TODO: Implement correction storage in database
    # Table: item_match_corrections
    # Fields: correction_id, user_input, barcode, corrected_item_id, 
    #         user_id, created_at
    
    logger.info(
        f"Human correction stored: '{user_input}' -> {corrected_item_id}"
    )
