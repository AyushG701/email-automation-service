from enum import Enum


class RMQEnum(Enum):
    EMAIL_INGESTION_ROUTING_KEY = 'email_ingestion_routing_key'
    EMAIL_INGESTION_QUEUE = 'email_ingestion_queue'
    EMAIL_INGESTION_EXCHANGE = 'email_ingestion_exchange'


class BIDDING_TYPE(Enum):
    STATUS_UPDATE = 'status_update',
    MANUAL = 'manual',
    AUTO = 'auto'


class LOAD_NEGOTIATION_DIRECTION(Enum):
    INCOMING = 'incoming'
    OUTGOING = 'outgoing'


class Load_Bid_Email_Intent(Enum):
    NEGOTIATION = 'negotiation'
    INFORMATION_SEEKING = 'information_seeking'
    BID_ACCEPTANCE = 'bid_acceptance'
    BID_REJECTION = 'bid_rejection'
    RATE_CONFIRMATION = 'rate_confirmation'
    BROKER_SETUP = 'broker_setup'


class EMAIL_CLASSIFICATION(Enum):
    LOAD = 'load'
    LOAD_OFFER = 'load_offer'
    LOAD_OFFER_NEGOTIATION = 'load_offer_bid'
    # LOAD_NEGOTIATION = 'load_negotiation'
    BROKER_SETUP = 'broker_setup'
