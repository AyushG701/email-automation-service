import logging
from typing import List, Dict, Any
from schemas.negotiation import NegotiationState
from services.multi_intent_extractor import MultiIntentExtractor, NegotiationIntent
from services.bid_calculator import FreightBidCalculator as BidCalculator
from services.information_seek import InformationSeeker
from core.utils import extract_price

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiIntentOrchestrator:
    """
    Orchestrates the negotiation process by managing state, classifying intent,
    and determining the next best action. This version is capable of handling
    multiple intents within a single email.
    """

    def __init__(self, state: NegotiationState, carrier_info: dict = None):
        self.state = state
        self.intent_extractor = MultiIntentExtractor()
        self.bid_calculator = BidCalculator()
        self.carrier_info = carrier_info or {}

    def process_email(self, email_body: str, chat_history: str) -> NegotiationState:
        """
        Main entry point for processing an incoming email.

        This method now handles multiple intents by processing them sequentially
        and updating the state after each action.
        """
        # 1. Extract all intents from the latest email
        intents = self._extract_intents(chat_history)
        logger.info(f"Extracted Intents (Multi-Intent Flow): {intents}")

        # Reset suggested response at the beginning of the turn
        self.state.suggested_response = ""

        # 2. Process each intent in the list
        for intent_data in intents:
            intent_name = intent_data.get("intent")
            intent_details = intent_data.get("details", {})

            handler = self._get_intent_handler(intent_name)
            self.state = handler(email_body, intent_details)

        # 3. Finalize the round
        self.state.last_message = email_body
        logger.info(f"Final State for this turn (Multi-Intent Flow): {self.state.dict()}")
        
        return self.state

    def _extract_intents(self, chat_history: str) -> List[Dict[str, Any]]:
        """Extracts a list of intents from the chat history."""
        return self.intent_extractor.extract_intents(chat_history)

    def _get_intent_handler(self, intent_name: str):
        """Maps an intent name to its corresponding handler method."""
        handlers = {
            NegotiationIntent.COUNTER_OFFER.value: self._handle_counter_offer,
            NegotiationIntent.ACCEPT_OFFER.value: self._handle_accept_offer,
            NegotiationIntent.REJECT_OFFER.value: self._handle_reject_offer,
            NegotiationIntent.ASK_FOR_INFO.value: self._handle_ask_for_info,
            NegotiationIntent.PROVIDE_INFO.value: self._handle_provide_info,
            NegotiationIntent.END_NEGOTIATION.value: self._handle_end_negotiation,
            NegotiationIntent.CONTINUE_NEGOTIATION.value: self._handle_continue_negotiation,
            NegotiationIntent.OTHER.value: self._handle_other,
        }
        return handlers.get(intent_name, self._handle_other)

    # --- Intent Handlers ---

    def _handle_counter_offer(self, email_body: str, details: Dict) -> NegotiationState:
        logger.info("Handling COUNTER_OFFER")
        self.state.negotiation_round += 1

        new_price = details.get("price") or extract_price(email_body)
        if new_price:
            self.state.last_broker_price = new_price
            logger.info(f"Extracted new price: ${new_price}")

        next_bid = self._calculate_next_bid()
        self.state.last_supertruck_price = next_bid
        self.state.response_needed = True
        response_text = f"Thank you for the offer. We can do ${next_bid}."
        self.state.suggested_response = f"{self.state.suggested_response}\n{response_text}".strip()

        return self.state

    def _handle_accept_offer(self, email_body: str, details: Dict) -> NegotiationState:
        logger.info("Handling ACCEPT_OFFER")
        self.state.is_negotiation_active = False
        self.state.response_needed = True
        response_text = "Great! We are happy to accept. Please send over the rate confirmation."
        self.state.suggested_response = f"{self.state.suggested_response}\n{response_text}".strip()
        return self.state

    def _handle_reject_offer(self, email_body: str, details: Dict) -> NegotiationState:
        logger.info("Handling REJECT_OFFER")
        next_bid = self._calculate_next_bid()
        self.state.last_supertruck_price = next_bid
        self.state.response_needed = True
        response_text = f"Understood. Our best offer is ${next_bid}. Let us know if that works."
        self.state.suggested_response = f"{self.state.suggested_response}\n{response_text}".strip()
        return self.state

    def _handle_ask_for_info(self, email_body: str, details: Dict) -> NegotiationState:
        logger.info("Handling ASK_FOR_INFO")
        question = details.get("question", email_body)
        answer = self._get_info_answer(question)
        self.state.info_exchange_round += 1
        self.state.response_needed = True
        self.state.suggested_response = f"{answer}\n\n{self.state.suggested_response}".strip()
        return self.state

    def _handle_provide_info(self, email_body: str, details: Dict) -> NegotiationState:
        logger.info("Handling PROVIDE_INFO")
        self.state.response_needed = False
        return self.state

    def _handle_end_negotiation(self, email_body: str, details: Dict) -> NegotiationState:
        logger.info("Handling END_NEGOTIATION")
        self.state.is_negotiation_active = False
        self.state.response_needed = False
        return self.state

    def _handle_continue_negotiation(self, email_body: str, details: Dict) -> NegotiationState:
        logger.info("Handling CONTINUE_NEGOTIATION")
        next_bid = self._calculate_next_bid()
        self.state.last_supertruck_price = next_bid
        self.state.response_needed = True
        response_text = f"To continue our discussion, our next offer is ${next_bid}."
        self.state.suggested_response = f"{self.state.suggested_response}\n{response_text}".strip()
        return self.state
        
    def _handle_other(self, email_body: str, details: Dict) -> NegotiationState:
        logger.info("Handling OTHER intent")
        self.state.response_needed = True
        response_text = "I'm not sure how to proceed. Could you clarify?"
        self.state.suggested_response = f"{self.state.suggested_response}\n{response_text}".strip()
        return self.state

    def _calculate_next_bid(self) -> float:
        """Calculate next bid based on current state using strategy logic."""
        min_price = getattr(self.state, 'strategy', None)
        if min_price and hasattr(min_price, 'min_price'):
            min_price = min_price.min_price
        else:
            min_price = 4000  # Default fallback

        last_broker = self.state.last_broker_price or 0
        last_ours = self.state.last_supertruck_price or self.state.initial_broker_price or 5000
        rounds = self.state.negotiation_round

        # Simple negotiation strategy
        if rounds <= 1:
            return last_ours  # Stay firm on first round
        elif rounds == 2:
            return last_ours * 0.95  # Drop 5%
        else:
            # Move toward middle but not below min
            mid_point = (last_broker + last_ours) / 2 if last_broker > 0 else last_ours * 0.9
            return max(mid_point, min_price)

    def _get_info_answer(self, question: str) -> str:
        """Get answer to info question using carrier data."""
        if not self.carrier_info:
            return "Let me check on that and get back to you. What's the load details?"

        try:
            from services.information_seek import InformationSeeker
            seeker = InformationSeeker(data=self.carrier_info)
            return seeker.ask(question=question)
        except Exception as e:
            logger.warning(f"Info seeker failed: {e}")
            return "Let me check on that. In the meantime, what rate are you looking for?"
