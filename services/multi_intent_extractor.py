"""
Enhanced Multi-Intent Extractor for Negotiation Communications
Version 2.0 - Production-Ready with Validation & Error Handling

Key Improvements:
1. ADDED: Comprehensive input validation
2. ADDED: Retry logic with exponential backoff
3. ADDED: Intent validation and normalization
4. ADDED: Confidence scoring for extracted intents
5. ADDED: Context-aware extraction with history analysis
6. ADDED: Pydantic models for type safety
7. ADDED: Smart caching for identical messages
8. ADDED: Detailed logging and error tracking
9. IMPROVED: Fallback strategies for API failures
10. IMPROVED: Cost optimization with token limits
"""

from enum import Enum
import json
import logging
import hashlib
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps, lru_cache
from pydantic import BaseModel, Field, field_validator, ValidationError
from core.openai import get_openai_client
from prompts.offer_negotiation import (
    NEGOTIATION_MULTI_INTENT_EXTRACTION_SYSTEM_PROMPT
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ExtractorConfig:
    """Configuration for the multi-intent extractor"""
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    MAX_CHAT_HISTORY_LENGTH = 8000  # tokens (approx 32k chars)
    MIN_CONFIDENCE_THRESHOLD = 0.3
    CACHE_SIZE = 100  # LRU cache size
    ENABLE_CACHING = True
    TEMPERATURE = 0.1
    MAX_TOKENS = 1000


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_with_backoff(max_retries=3, initial_delay=1.0):
    """Decorator for retrying API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Max retries reached. Last error: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# ENUMS
# =============================================================================

class NegotiationIntent(str, Enum):
    """
    Enhanced enum for classifying intents in a negotiation context.
    Each intent represents a distinct action or request.
    """
    # Primary negotiation actions
    ACCEPT_OFFER = "ACCEPT_OFFER"
    REJECT_OFFER = "REJECT_OFFER"
    COUNTER_OFFER = "COUNTER_OFFER"
    
    # Information exchange
    ASK_FOR_INFO = "ASK_FOR_INFO"
    PROVIDE_INFO = "PROVIDE_INFO"
    
    # Negotiation flow control
    END_NEGOTIATION = "END_NEGOTIATION"
    CONTINUE_NEGOTIATION = "CONTINUE_NEGOTIATION"
    
    # Additional common intents
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    CONFIRM_DETAILS = "CONFIRM_DETAILS"
    EXPRESS_INTEREST = "EXPRESS_INTEREST"
    
    # Fallback
    OTHER = "OTHER"


class ConfidenceLevel(str, Enum):
    """Confidence levels for extracted intents"""
    HIGH = "high"      # > 0.7
    MEDIUM = "medium"  # 0.4 - 0.7
    LOW = "low"        # < 0.4


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ExtractedIntent(BaseModel):
    """
    Model representing a single extracted intent with validation.
    """
    intent: str = Field(..., description="The classified intent")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the intent"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why this intent was identified"
    )
    
    @field_validator('intent', mode='before')
    @classmethod
    def validate_intent(cls, v):
        """Validate and normalize intent values"""
        if not v:
            return NegotiationIntent.OTHER.value
        
        v_upper = str(v).upper().strip()
        
        # Try exact match first
        valid_intents = [intent.value for intent in NegotiationIntent]
        if v_upper in valid_intents:
            return v_upper
        
        # Try fuzzy matching for common variations
        intent_aliases = {
            "ACCEPT": NegotiationIntent.ACCEPT_OFFER.value,
            "REJECT": NegotiationIntent.REJECT_OFFER.value,
            "COUNTER": NegotiationIntent.COUNTER_OFFER.value,
            "ASK": NegotiationIntent.ASK_FOR_INFO.value,
            "PROVIDE": NegotiationIntent.PROVIDE_INFO.value,
            "QUESTION": NegotiationIntent.ASK_FOR_INFO.value,
            "ANSWER": NegotiationIntent.PROVIDE_INFO.value,
            "END": NegotiationIntent.END_NEGOTIATION.value,
            "CONTINUE": NegotiationIntent.CONTINUE_NEGOTIATION.value,
        }
        
        for alias, intent in intent_aliases.items():
            if alias in v_upper:
                return intent
        
        logger.warning(f"Unknown intent '{v}', defaulting to OTHER")
        return NegotiationIntent.OTHER.value
    
    @field_validator('confidence', mode='before')
    @classmethod
    def normalize_confidence(cls, v):
        """Normalize confidence to 0-1 range"""
        if v is None:
            return 0.5
        try:
            v = float(v)
            return max(0.0, min(1.0, v))
        except (ValueError, TypeError):
            return 0.5
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get categorical confidence level"""
        if self.confidence > 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence > 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class IntentExtractionResult(BaseModel):
    """
    Complete result of intent extraction with metadata.
    """
    intents: List[ExtractedIntent] = Field(
        default_factory=list,
        description="List of extracted intents"
    )
    message_summary: Optional[str] = Field(
        None,
        description="Brief summary of the message"
    )
    extraction_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the extraction process"
    )
    
    def has_high_confidence_intents(self) -> bool:
        """Check if any intents have high confidence"""
        return any(
            intent.get_confidence_level() == ConfidenceLevel.HIGH 
            for intent in self.intents
        )
    
    def get_primary_intent(self) -> Optional[ExtractedIntent]:
        """Get the intent with highest confidence"""
        if not self.intents:
            return None
        return max(self.intents, key=lambda x: x.confidence)
    
    def filter_by_confidence(self, min_confidence: float) -> List[ExtractedIntent]:
        """Filter intents by minimum confidence threshold"""
        return [
            intent for intent in self.intents 
            if intent.confidence >= min_confidence
        ]


# =============================================================================
# CHAT HISTORY PREPROCESSOR
# =============================================================================

class ChatHistoryPreprocessor:
    """Preprocesses and validates chat history"""
    
    @staticmethod
    def validate_and_truncate(
        chat_history: str, 
        max_length: int = ExtractorConfig.MAX_CHAT_HISTORY_LENGTH
    ) -> str:
        """
        Validate and truncate chat history to reasonable length.
        Prioritizes recent messages.
        """
        if not chat_history or not chat_history.strip():
            raise ValueError("Chat history is empty")
        
        chat_history = chat_history.strip()
        
        # If within limit, return as-is
        if len(chat_history) <= max_length:
            return chat_history
        
        # Truncate, keeping the most recent messages
        logger.warning(
            f"Chat history too long ({len(chat_history)} chars), "
            f"truncating to {max_length} chars"
        )
        
        # Try to find a reasonable break point (paragraph or newline)
        truncated = chat_history[-max_length:]
        
        # Find first newline to avoid cutting mid-sentence
        first_newline = truncated.find('\n')
        if first_newline > 0 and first_newline < 500:
            truncated = truncated[first_newline:].strip()
        
        return f"[...earlier messages truncated...]\n\n{truncated}"
    
    @staticmethod
    def extract_last_message(chat_history: str) -> str:
        """Extract the last message from chat history"""
        # Simple heuristic: split by double newline or common separators
        messages = chat_history.strip().split('\n\n')
        if messages:
            return messages[-1].strip()
        return chat_history.strip()
    
    @staticmethod
    def get_message_hash(chat_history: str) -> str:
        """Generate hash for caching purposes"""
        return hashlib.md5(chat_history.encode()).hexdigest()


# =============================================================================
# RESPONSE PARSER
# =============================================================================

class ResponseParser:
    """Parses and validates LLM responses"""
    
    @staticmethod
    def parse_intent_response(response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response with multiple fallback strategies.
        """
        if not response or not response.strip():
            raise ValueError("Empty response from LLM")
        
        response = response.strip()
        
        # Strategy 1: Direct JSON parse
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # If single intent returned as dict, wrap in list
                return [data]
            else:
                raise ValueError(f"Unexpected response type: {type(data)}")
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from code block
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                data = json.loads(code_block_match.group(1).strip())
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON array or object
        json_match = re.search(r'(\[.*\]|\{.*\})', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse response as JSON: {response[:200]}")
    
    @staticmethod
    def validate_and_normalize_intents(
        raw_intents: List[Dict[str, Any]]
    ) -> List[ExtractedIntent]:
        """
        Validate and normalize extracted intents using Pydantic.
        """
        validated_intents = []
        
        for i, raw_intent in enumerate(raw_intents):
            try:
                # Ensure required fields exist
                if 'intent' not in raw_intent:
                    logger.warning(f"Intent #{i} missing 'intent' field, skipping")
                    continue
                
                # Add defaults for optional fields
                if 'confidence' not in raw_intent:
                    raw_intent['confidence'] = 0.5
                if 'details' not in raw_intent:
                    raw_intent['details'] = {}
                
                # Validate with Pydantic
                validated = ExtractedIntent(**raw_intent)
                validated_intents.append(validated)
                
            except ValidationError as e:
                logger.warning(f"Validation failed for intent #{i}: {e}")
                # Create fallback intent
                validated_intents.append(ExtractedIntent(
                    intent=NegotiationIntent.OTHER.value,
                    confidence=0.3,
                    details={"error": str(e), "raw_data": raw_intent},
                    reasoning="Failed validation"
                ))
        
        return validated_intents


# =============================================================================
# MAIN EXTRACTOR SERVICE
# =============================================================================

class MultiIntentExtractor:
    """
    Enhanced multi-intent extractor with validation, retry logic, and caching.
    """
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.openai_client = get_openai_client()
        self.config = config or ExtractorConfig()
        self.preprocessor = ChatHistoryPreprocessor()
        self.parser = ResponseParser()
        self.logger = logging.getLogger(__name__)
        
        # Simple in-memory cache
        self._cache = {} if self.config.ENABLE_CACHING else None
    
    def _get_from_cache(self, message_hash: str) -> Optional[IntentExtractionResult]:
        """Get cached result if available"""
        if not self._cache:
            return None
        return self._cache.get(message_hash)
    
    def _save_to_cache(self, message_hash: str, result: IntentExtractionResult):
        """Save result to cache"""
        if not self._cache:
            return
        
        # Implement simple LRU by limiting cache size
        if len(self._cache) >= self.config.CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[message_hash] = result
    
    @retry_with_backoff(
        max_retries=ExtractorConfig.MAX_RETRIES,
        initial_delay=ExtractorConfig.RETRY_DELAY
    )
    def _call_llm(self, chat_history: str) -> str:
        """Call LLM with retry logic"""
        response = self.openai_client.chat_completion(
            system_prompt=NEGOTIATION_MULTI_INTENT_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=chat_history,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS
        )
        return response
    
    def extract_intents(
        self, 
        chat_history: str,
        use_cache: bool = True
    ) -> IntentExtractionResult:
        """
        Analyzes the chat history to extract a list of intents from the last message.

        Args:
            chat_history: The full conversation history
            use_cache: Whether to use cached results

        Returns:
            IntentExtractionResult with list of validated intents and metadata
        """
        extraction_start = time.time()
        
        try:
            # Step 1: Validate and preprocess input
            chat_history = self.preprocessor.validate_and_truncate(chat_history)
            message_hash = self.preprocessor.get_message_hash(chat_history)
            
            # Step 2: Check cache
            if use_cache:
                cached_result = self._get_from_cache(message_hash)
                if cached_result:
                    self.logger.info("Returning cached result")
                    return cached_result
            
            # Step 3: Call LLM
            self.logger.info("Calling LLM for intent extraction")
            raw_response = self._call_llm(chat_history)
            
            # Step 4: Parse response
            raw_intents = self.parser.parse_intent_response(raw_response)
            
            # Step 5: Validate and normalize
            validated_intents = self.parser.validate_and_normalize_intents(raw_intents)
            
            # Step 6: Filter low-confidence intents
            filtered_intents = [
                intent for intent in validated_intents
                if intent.confidence >= self.config.MIN_CONFIDENCE_THRESHOLD
            ]
            
            if not filtered_intents and validated_intents:
                self.logger.warning(
                    f"All intents below confidence threshold "
                    f"({self.config.MIN_CONFIDENCE_THRESHOLD}), keeping best one"
                )
                filtered_intents = [max(validated_intents, key=lambda x: x.confidence)]
            
            # Step 7: Build result
            extraction_time = time.time() - extraction_start
            last_message = self.preprocessor.extract_last_message(chat_history)
            
            result = IntentExtractionResult(
                intents=filtered_intents,
                message_summary=last_message[:200] + "..." if len(last_message) > 200 else last_message,
                extraction_metadata={
                    "extraction_time_seconds": round(extraction_time, 3),
                    "total_intents_found": len(raw_intents),
                    "intents_after_filtering": len(filtered_intents),
                    "message_hash": message_hash,
                    "chat_history_length": len(chat_history),
                }
            )
            
            # Step 8: Cache result
            if use_cache:
                self._save_to_cache(message_hash, result)
            
            self.logger.info(
                f"Extracted {len(filtered_intents)} intents in {extraction_time:.2f}s"
            )
            
            return result
            
        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            return self._create_fallback_result(str(e), "validation_error")
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return self._create_fallback_result(str(e), "parsing_error")
        
        except Exception as e:
            self.logger.error(f"Unexpected error in intent extraction: {e}", exc_info=True)
            return self._create_fallback_result(str(e), "unexpected_error")
    
    def _create_fallback_result(
        self, 
        error_message: str,
        error_type: str
    ) -> IntentExtractionResult:
        """Create fallback result when extraction fails"""
        return IntentExtractionResult(
            intents=[
                ExtractedIntent(
                    intent=NegotiationIntent.OTHER.value,
                    confidence=0.1,
                    details={
                        "error": error_message,
                        "error_type": error_type
                    },
                    reasoning="Intent extraction failed, manual review required"
                )
            ],
            extraction_metadata={
                "error": error_message,
                "error_type": error_type,
                "fallback": True
            }
        )
    
    def extract_intents_legacy(self, chat_history: str) -> List[Dict[str, Any]]:
        """
        Legacy method for backwards compatibility.
        Returns raw list of dictionaries instead of IntentExtractionResult.
        """
        result = self.extract_intents(chat_history)
        return [intent.model_dump() for intent in result.intents]


# =============================================================================
# ENHANCED TEST SUITE
# =============================================================================

def run_tests():
    """Run comprehensive test cases"""
    test_cases = [
        {
            "chat_history": """
            Broker: Can you haul this load from Dallas to Houston for $1500?
            Carrier: I can do it for $1800.
            """,
            "expected_intents": ["COUNTER_OFFER"],
            "description": "Simple counter-offer"
        },
        {
            "chat_history": """
            Broker: Can you do $1600?
            Carrier: Yes, that works. What's your MC number and payment terms?
            """,
            "expected_intents": ["ACCEPT_OFFER", "ASK_FOR_INFO"],
            "description": "Accept + ask for info (multiple intents)"
        },
        {
            "chat_history": """
            Broker: Our MC is 123456, payment is NET 30.
            """,
            "expected_intents": ["PROVIDE_INFO"],
            "description": "Providing information"
        },
        {
            "chat_history": """
            Carrier: Sorry, can't do it for less than $2000.
            """,
            "expected_intents": ["REJECT_OFFER"],
            "description": "Rejection"
        },
        {
            "chat_history": """
            Broker: What's the weight and equipment type?
            Carrier: 40,000 lbs, need a 53' dry van
            """,
            "expected_intents": ["PROVIDE_INFO"],
            "description": "Answering question"
        },
        {
            "chat_history": "",
            "expected_intents": ["OTHER"],
            "description": "Empty chat history (fallback)"
        },
    ]
    
    print("=" * 80)
    print("MULTI-INTENT EXTRACTOR TESTS (v2.0)")
    print("=" * 80)
    
    extractor = MultiIntentExtractor()
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_cases)}: {test['description']}")
        print(f"{'='*80}")
        print(f"Chat: {test['chat_history'][:100]}{'...' if len(test['chat_history']) > 100 else ''}")
        
        try:
            result = extractor.extract_intents(test['chat_history'])
            
            extracted_intent_types = [intent.intent for intent in result.intents]
            
            # Check if any expected intent is present
            has_match = any(
                exp in extracted_intent_types 
                for exp in test['expected_intents']
            )
            
            status = "✅ PASS" if has_match else "❌ FAIL"
            
            print(f"Expected: {test['expected_intents']}")
            print(f"Got:      {extracted_intent_types}")
            print(f"Confidences: {[f'{i.confidence:.2f}' for i in result.intents]}")
            print(f"\nStatus: {status}")
            
            if has_match:
                passed += 1
            else:
                failed += 1
            
            results.append(has_match)
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            failed += 1
            results.append(False)
    
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS: {passed}/{len(test_cases)} passed, {failed} failed")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print("=" * 80)
    
    return all(results)


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)