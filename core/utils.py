import re
from typing import Optional

def extract_price(text: str) -> Optional[float]:
    """
    Extracts a price from a string, handling formats like "$1,500", "2k", "$2.5k".
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    # Handle 'k' notation first, as it's more specific.
    # Handles "$2.5k", "2.5k", "2k"
    k_pattern = r'\$?\s*([\d,]+(?:\.\d+)?)\s*k\b'
    k_matches = re.findall(k_pattern, text_lower)
    for match in k_matches:
        try:
            value = float(match.replace(',', '')) * 1000
            # Using a reasonable price range check from original code
            if 300 <= value <= 50000:
                return value
        except (ValueError, TypeError):
            continue

    # Original patterns from PriceExtractor, avoiding 'k' which is now handled
    PRICE_PATTERNS = [
        r'\$\s*([\d,]+(?:\.\d{2})?)',      # $1,234.56
        r'([\d,]+)\s*(?:dollars?|usd)',   # 1234 dollars
        r'(?:^|\s)([\d]{4,5})(?:\s|$|\.)', # 1234 (as a standalone number)
    ]

    for pattern in PRICE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match.replace(',', ''))
                if 300 <= value <= 50000:
                    return value
            except (ValueError, TypeError):
                continue

    return None
