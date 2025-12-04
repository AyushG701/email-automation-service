from typing import Union
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import aio_pika
import asyncio
import json
import httpx
from contextlib import asynccontextmanager

from core.rabbitmq import RabbitMQManager
from core.auth_client import AuthApiClient

from models.email import EmailMessage
from constant.enum import RMQEnum, EMAIL_CLASSIFICATION
from config import AppConfig

from services.email_classifier import EmailClassifierOpenAI
from services.conversation import InformationSeeker

from integration.supertruck import SuperTruck
from integration.load_board import LoadBoard

# from actions.load_offer import LoadOfferAction
# from actions.load import LoadAction
from actions.negotiation import NegotiationAction

from api.v1.negotiation import NegotiationController
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()


config = AppConfig()
# ------------------------------
# Create RabbitMQ manager
# ------------------------------
rabbit_mq_manager = RabbitMQManager(
    rabbitmq_url=config.RABBITMQ_URL,
    queue_name=RMQEnum.EMAIL_INGESTION_QUEUE.value
)


# ------------------------------
# Lifespan (startup / shutdown)
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Attempting RabbitMQ connection...")
    # Startup
    await rabbit_mq_manager.connect()
    consumer_task = asyncio.create_task(
        rabbit_mq_manager.consume(message_processor)
    )
    app.state.consumer_task = consumer_task
    logger.info("Connected")
    yield

    # Shutdown
    rabbit_mq_manager.should_reconnect = False

    if hasattr(app.state, "consumer_task"):
        app.state.consumer_task.cancel()
        try:
            await app.state.consumer_task
        except asyncio.CancelledError:
            pass

    await rabbit_mq_manager.disconnect()


# ------------------------------
# FastAPI instance (SINGLE)
# ------------------------------
app = FastAPI(
    title="SuperTruck Email Automation",
    description="APIs for email processing and negotiation",
    version="0.1.0",
    root_path="/email-processor",
    lifespan=lifespan
)


# ------------------------------
# Middleware
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
# Routers
# ------------------------------
negotiation_router = NegotiationController().router
app.include_router(negotiation_router)

received_messages = []


supertruck = SuperTruck()
negotiation = NegotiationAction()
load_board = LoadBoard()


def remove_none_values(data: dict) -> dict:
    return {k: v for k, v in data.items() if v is not None}


# *MAIN PROCESSOR THAT EXECUTES/DETERMINES THE NEXT ACTION
async def message_processor(message: aio_pika.IncomingMessage):
    logger.info("🔥 MESSAGE RECEIVED IN PROCESSOR!")
    async with message.process():
        try:
            logger.info("Message is here........")
            classification = ""
            email_data_decode = message.body.decode()
            email_data = json.loads(email_data_decode)

            # !NEED TO CHANGE FOR DEVELOPMENT PURPOSE ONLY
            received_messages.append(json.loads(email_data_decode))
            # print(f"Received and processing email: {email_data}")
            params = {
                "tenant_id": email_data["tenantId"],
                "to": email_data["to"],
                "from": email_data["from"],
                "thread_Id": email_data["threadId"],
                "body": email_data["body"],
                "messageId": email_data["messageId"],
                "inReplyTo": email_data["inReplyTo"],
                "references": email_data["references"]
            }
            logger.info(email_data)
            # * FIND PREV CONVERSATION FROM EMAIL LOGS AND CONTINUE CONVERSATION OR CLASSIFY EMAIL
            if (email_data["inReplyTo"]):
                logger.info(f"Continue from reply")
                email_log = await supertruck.find_email_log(
                    tenant_id=email_data["tenantId"], thread_id=email_data["threadId"])
                await negotiation.execute_negotiation(data=params)
            else:
                # * OPENAI EMAIL CLASSIFICATION
                if not any(email_data.get(key) for key in ["subject"]):
                    raise ValueError(
                        "No relevant email content (Subject, Short Preview, or Main Body (Trimmed)) provided.")
                # information_seeker.ask()
                classifier = EmailClassifierOpenAI()
                output = classifier.process_email(email_data)
                # print(f"Email classification output:{output}")

                classification = output["classification"]
                cleaned_data = remove_none_values(
                    output["extracted_details"])
                logger.info(
                    f"Email data classified & cleaned successfully")
                # * INTEGRATION ACCORDING TO CLASSIFICATION
                actions = {
                    # EMAIL_CLASSIFICATION.LOAD_NEGOTIATION.value: lambda: loadNegotiation.execute_load_negotiation(data=params),
                    EMAIL_CLASSIFICATION.LOAD_OFFER_NEGOTIATION.value: lambda: negotiation.execute_negotiation(data=params),
                    EMAIL_CLASSIFICATION.LOAD_OFFER.value: lambda: load_board.create_load_offer(
                        tenant_id=email_data["tenantId"], data=cleaned_data),
                    EMAIL_CLASSIFICATION.LOAD.value: lambda: load_board.create_load(
                        tenant_id=email_data["tenantId"], data=output["extracted_details"]
                    ),
                    EMAIL_CLASSIFICATION.BROKER_SETUP.value: lambda: print("Broker setup"),
                }

                action = actions.get(
                    classification, lambda: print("Default classification"))
                await action()

        except httpx.HTTPStatusError as e:
            print("HTTP error:", e)
            return e

        except ValueError as e:
            print("Parsing/validation error:", e)
            return e
            # raise HTTPException(status_code=400, detail=str(e))

        except KeyError as e:
            print("Missing field error:", e)
            return e
            # raise HTTPException(status_code=500, detail=f"Key error: {e}")

        except Exception as e:
            print("Unexpected error:", repr(e))
            return e
            # raise HTTPException(status_code=500, detail=str(e))


# !NEED TO CHANGE FOR DEVELOPMENT PURPOSE ONLY
@app.get("/email")
async def receive_message():
    return {"messages": received_messages}


@app.post("/send")
async def send_message(
    message: EmailMessage = Body(
        ...,
        example=EmailMessage.Config.json_schema_extra.get("example")
        if hasattr(EmailMessage, 'Config') and hasattr(EmailMessage.Config, 'json_schema_extra')
        else None
    )
):
    """
    Send an email message to the RabbitMQ queue for processing.

    - **message**: EmailMessage object with all email details
    """
    try:
        # Convert Pydantic model to dict
        message_dict = message.dict(by_alias=True)

        # Publish to RabbitMQ
        await rabbit_mq_manager.publish(message_dict)

        logger.info(
            f"✅ Message published successfully: {message_dict.get('subject', 'No subject')}")

        return {
            "status": "success",
            "message": "Email message sent to queue",
            "queue": rabbit_mq_manager.queue_name,
            "data": message_dict
        }

    except Exception as e:
        logger.error(f"❌ Failed to send message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send message to RabbitMQ: {str(e)}"
        )


@app.get("/")
def read_root():
    return {"Hello": "Developer"}
