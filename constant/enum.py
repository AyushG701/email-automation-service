"""
Enums for Email Automation Service
Updated with non-overlapping intent categories
"""
from enum import Enum


class RMQEnum(Enum):
    EMAIL_INGESTION_ROUTING_KEY = 'email_ingestion_routing_key'
    EMAIL_INGESTION_QUEUE = 'email_ingestion_queue'
    EMAIL_INGESTION_EXCHANGE = 'email_ingestion_exchange'


class BIDDING_TYPE(Enum):
    STATUS_UPDATE = 'status_update'
    MANUAL = 'manual'
    AUTO = 'auto'


class LOAD_NEGOTIATION_DIRECTION(Enum):
    INCOMING = 'incoming'
    OUTGOING = 'outgoing'


class Load_Bid_Email_Intent(Enum):
    """
    Email intent categories - Updated for non-overlapping classification.

    KEY CHANGES:
    - Price-related intents are clearly separated from info-related
    - Only PRICE intents affect negotiation round counting
    - INFO intents do NOT affect negotiation rounds
    """
    # Price-related intents (affect negotiation rounds)
    NEGOTIATION = 'negotiation'           # Active price discussion
    BID_ACCEPTANCE = 'bid_acceptance'     # Broker accepts price
    BID_REJECTION = 'bid_rejection'       # Broker rejects price

    # Info-related intents (do NOT affect negotiation rounds)
    INFORMATION_SEEKING = 'information_seeking'  # Broker asks questions
    LOAD_DETAILS = 'load_details'         # Broker provides load info

    # Admin intents
    RATE_CONFIRMATION = 'rate_confirmation'  # RC document
    BROKER_SETUP = 'broker_setup'            # Carrier setup request
    BOOKING_REQUEST = 'booking_request'      # Ready to book

    # Fallback
    UNCLEAR = 'unclear'


class EMAIL_CLASSIFICATION(Enum):
    """Email classification categories for initial email processing"""
    LOAD = 'load'
    LOAD_OFFER = 'load_offer'
    LOAD_OFFER_NEGOTIATION = 'load_offer_bid'
    BROKER_SETUP = 'broker_setup'


# =============================================================================
# NEW: Conversation State Enum for State Machine
# =============================================================================

class ConversationState(Enum):
    """State machine states for conversation flow"""
    INITIAL = 'initial'
    INFO_GATHERING = 'info_gathering'
    READY_TO_NEGOTIATE = 'ready_to_negotiate'
    NEGOTIATING = 'negotiating'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    STALLED = 'stalled'


# =============================================================================
# NEW: Message Intent Enum (Non-overlapping)
# =============================================================================

class MessageIntent(Enum):
    """
    Non-overlapping intent categories for accurate classification.

    These are used internally by the NegotiationOrchestrator.
    """
    # Price-related (affects negotiation rounds)
    PRICE_OFFER = 'price_offer'           # Broker offers a price
    PRICE_COUNTER = 'price_counter'       # Counter-offer on price
    PRICE_INQUIRY = 'price_inquiry'       # Asking for a rate
    PRICE_ACCEPTANCE = 'price_acceptance' # Accepting a price
    PRICE_REJECTION = 'price_rejection'   # Rejecting a price

    # Info-related (does NOT affect negotiation rounds)
    INFO_REQUEST = 'info_request'         # Asking for info
    INFO_RESPONSE = 'info_response'       # Providing info

    # Admin
    RATE_CONFIRMATION = 'rate_confirmation'
    BOOKING_REQUEST = 'booking_request'
    BROKER_SETUP = 'broker_setup'

    # Fallback
    UNCLEAR = 'unclear'


# =============================================================================
# Intent Mapping (Legacy to New)
# =============================================================================

def map_legacy_intent_to_new(legacy_intent: str) -> MessageIntent:
    """Map legacy intent categories to new non-overlapping ones"""
    mapping = {
        'negotiation': MessageIntent.PRICE_COUNTER,
        'bid_acceptance': MessageIntent.PRICE_ACCEPTANCE,
        'bid_rejection': MessageIntent.PRICE_REJECTION,
        'information_seeking': MessageIntent.INFO_REQUEST,
        'load_details': MessageIntent.INFO_RESPONSE,
        'rate_confirmation': MessageIntent.RATE_CONFIRMATION,
        'broker_setup': MessageIntent.BROKER_SETUP,
        'booking_request': MessageIntent.BOOKING_REQUEST,
        'unclear': MessageIntent.UNCLEAR,
    }
    return mapping.get(legacy_intent.lower(), MessageIntent.UNCLEAR)


def map_new_intent_to_legacy(new_intent: MessageIntent) -> str:
    """Map new intent categories to legacy ones for backwards compatibility"""
    mapping = {
        MessageIntent.PRICE_OFFER: 'negotiation',
        MessageIntent.PRICE_COUNTER: 'negotiation',
        MessageIntent.PRICE_INQUIRY: 'negotiation',
        MessageIntent.PRICE_ACCEPTANCE: 'bid_acceptance',
        MessageIntent.PRICE_REJECTION: 'bid_rejection',
        MessageIntent.INFO_REQUEST: 'information_seeking',
        MessageIntent.INFO_RESPONSE: 'load_details',
        MessageIntent.RATE_CONFIRMATION: 'rate_confirmation',
        MessageIntent.BROKER_SETUP: 'broker_setup',
        MessageIntent.BOOKING_REQUEST: 'booking_request',
        MessageIntent.UNCLEAR: 'unclear',
    }
    return mapping.get(new_intent, 'unclear')
