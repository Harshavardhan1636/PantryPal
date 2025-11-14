"""
Text normalization utilities for item canonicalization.

Preprocessing pipeline for cleaning and standardizing free text, OCR results,
and user input before matching against items_catalog.
"""

import re
import unicodedata
from typing import List, Optional, Set
from dataclasses import dataclass


# Common stopwords for food items
FOOD_STOPWORDS: Set[str] = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for",
    "with", "from", "by", "about", "as", "into", "like", "through",
    "organic", "fresh", "frozen", "canned", "whole", "natural", "pure",
    "100%", "premium", "deluxe", "choice", "select", "best", "great",
    "value", "pack", "family", "size", "oz", "lb", "g", "kg", "ml", "l",
}


# Quantity unit patterns to remove
QUANTITY_PATTERNS: List[str] = [
    r'\d+\s*(oz|ounces?)',
    r'\d+\s*(lb|lbs|pounds?)',
    r'\d+\s*(g|grams?)',
    r'\d+\s*(kg|kilograms?)',
    r'\d+\s*(ml|milliliters?)',
    r'\d+\s*(l|liters?)',
    r'\d+\s*(count|ct|pack|pk)',
    r'\d+\s*x\s*\d+',
    r'\d+[-/]\d+',
]


# Brand/descriptor patterns that can be removed
DESCRIPTOR_PATTERNS: List[str] = [
    r'\b(organic|natural|fresh|frozen|dried)\b',
    r'\b(whole|sliced|diced|chopped|minced)\b',
    r'\b(low|reduced|free|no)\s+(fat|sodium|sugar|calorie)',
    r'\b(extra|super|ultra)\s+(large|small)',
]


@dataclass
class NormalizedText:
    """Result of text normalization."""
    original: str
    normalized: str
    removed_tokens: List[str]
    confidence_adjustment: float  # Penalty for heavy modifications


class TextNormalizer:
    """Text normalization pipeline for item matching."""
    
    def __init__(
        self,
        remove_stopwords: bool = True,
        remove_quantities: bool = True,
        remove_descriptors: bool = True,
        lowercase: bool = True,
        remove_diacritics: bool = True,
    ):
        self.remove_stopwords = remove_stopwords
        self.remove_quantities = remove_quantities
        self.remove_descriptors = remove_descriptors
        self.lowercase = lowercase
        self.remove_diacritics = remove_diacritics
        
        # Compile regex patterns
        self.quantity_regex = re.compile(
            "|".join(QUANTITY_PATTERNS),
            re.IGNORECASE
        )
        self.descriptor_regex = re.compile(
            "|".join(DESCRIPTOR_PATTERNS),
            re.IGNORECASE
        )
    
    def normalize(self, text: str) -> NormalizedText:
        """
        Apply full normalization pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            NormalizedText with original, normalized, and metadata
        """
        if not text or not text.strip():
            return NormalizedText(
                original=text,
                normalized="",
                removed_tokens=[],
                confidence_adjustment=0.0,
            )
        
        original = text
        removed_tokens = []
        modification_count = 0
        
        # Step 1: Unicode normalization
        if self.remove_diacritics:
            text = self._remove_diacritics(text)
            modification_count += 1
        
        # Step 2: Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Step 3: Remove quantity patterns
        if self.remove_quantities:
            matches = self.quantity_regex.findall(text)
            if matches:
                removed_tokens.extend(matches)
                text = self.quantity_regex.sub(" ", text)
                modification_count += len(matches)
        
        # Step 4: Remove descriptor patterns
        if self.remove_descriptors:
            matches = self.descriptor_regex.findall(text)
            if matches:
                removed_tokens.extend(matches)
                text = self.descriptor_regex.sub(" ", text)
                modification_count += len(matches)
        
        # Step 5: Remove special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Step 6: Remove stopwords
        if self.remove_stopwords:
            tokens = text.split()
            original_token_count = len(tokens)
            tokens = [t for t in tokens if t not in FOOD_STOPWORDS]
            removed_count = original_token_count - len(tokens)
            if removed_count > 0:
                modification_count += removed_count
            text = " ".join(tokens)
        
        # Step 7: Normalize whitespace
        text = " ".join(text.split())
        
        # Calculate confidence adjustment penalty
        # Heavy modifications reduce confidence (0.0 = no penalty, -1.0 = max penalty)
        confidence_adjustment = -min(0.3, modification_count * 0.05)
        
        return NormalizedText(
            original=original,
            normalized=text,
            removed_tokens=removed_tokens,
            confidence_adjustment=confidence_adjustment,
        )
    
    def _remove_diacritics(self, text: str) -> str:
        """
        Remove diacritical marks (accents) from unicode text.
        
        Example: "café" -> "cafe"
        """
        nfkd_form = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    def normalize_batch(self, texts: List[str]) -> List[NormalizedText]:
        """Normalize multiple texts efficiently."""
        return [self.normalize(text) for text in texts]


def extract_brand(text: str) -> Optional[str]:
    """
    Extract brand name from product text (simple heuristic).
    
    Example: "Great Value Organic Milk" -> "Great Value"
    """
    # Brand is often first 1-2 capitalized words
    words = text.split()
    brand_words = []
    
    for word in words[:3]:  # Check first 3 words
        if word and word[0].isupper():
            brand_words.append(word)
        else:
            break
    
    return " ".join(brand_words) if brand_words else None


def extract_core_name(text: str) -> str:
    """
    Extract core item name by removing brand and descriptors.
    
    Example: "Great Value Organic 2% Milk 1 Gallon" -> "milk"
    """
    normalizer = TextNormalizer(
        remove_stopwords=True,
        remove_quantities=True,
        remove_descriptors=True,
    )
    result = normalizer.normalize(text)
    
    # Return last significant word (usually the core item)
    tokens = result.normalized.split()
    if tokens:
        return tokens[-1] if len(tokens) == 1 else " ".join(tokens[-2:])
    return ""


def similarity_score(text1: str, text2: str) -> float:
    """
    Calculate simple character-level similarity (Jaccard).
    
    Returns:
        Float between 0 and 1 (1 = identical)
    """
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    normalizer = TextNormalizer()
    
    test_cases = [
        "Great Value Organic 2% Milk 1 Gallon",
        "Fresh Whole Sliced Strawberries 16 oz",
        "Café Bustelo Espresso Coffee 10oz Can",
        "Lay's Classic Potato Chips Family Size 13oz",
        "365 by Whole Foods Market Quinoa Organic 16 Ounces",
    ]
    
    print("Text Normalization Examples:")
    print("=" * 80)
    for text in test_cases:
        result = normalizer.normalize(text)
        brand = extract_brand(text)
        core = extract_core_name(text)
        
        print(f"\nOriginal:   {result.original}")
        print(f"Normalized: {result.normalized}")
        print(f"Brand:      {brand}")
        print(f"Core Name:  {core}")
        print(f"Removed:    {', '.join(result.removed_tokens[:3])}...")
        print(f"Confidence: {1.0 + result.confidence_adjustment:.2f}")
