import asyncio
import aio_pika
import json
import logging

logger = logging.getLogger(__name__)


class RabbitMQManager:
    def __init__(self, rabbitmq_url: str, queue_name: str):
        self.rabbitmq_url = rabbitmq_url
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self.should_reconnect = True
        self.consumer_tag = None

    async def connect(self):
        """Connect and create channel with auto-reconnect"""
        while self.should_reconnect:
            try:
                logger.info(f"Connecting to RabbitMQ: {self.rabbitmq_url}")
                self.connection = await aio_pika.connect_robust(
                    self.rabbitmq_url,
                    heartbeat=60,
                    connection_attempts=3,
                    retry_delay=5
                )
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=1)

                queue = await self.channel.declare_queue(
                    self.queue_name,
                    durable=True
                )

                logger.info(f"✅ Connected to RabbitMQ & channel ready")
                logger.info(
                    f"📊 Queue '{self.queue_name}' has {queue.declaration_result.message_count} messages")
                return

            except Exception as e:
                logger.error(f"❌ Having trouble connecting. Will try again in 5 seconds... Error: {e}")
                await asyncio.sleep(5)

    async def disconnect(self):
        """Graceful shutdown"""
        logger.info("Disconnecting from RabbitMQ...")
        self.should_reconnect = False

        if self.channel and not self.channel.is_closed:
            await self.channel.close()
        if self.connection and not self.connection.is_closed:
            await self.connection.close()

        logger.info("🔌 Disconnected from RabbitMQ")

    async def publish(self, message: dict):
        """Safe publish with auto-reconnect"""
        if not self.channel or self.channel.is_closed:
            logger.warning(
                "⚠️ Channel closed — reconnecting before publish...")
            await self.connect()

        msg_str = json.dumps(message)

        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=msg_str.encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=self.queue_name,
        )

        logger.info(f"📤 Message sent to queue: {self.queue_name}")

    async def consume(self, callback):
        """Consume with reconnection behavior - FIXED VERSION"""
        while self.should_reconnect:
            try:
                if not self.channel or self.channel.is_closed:
                    await self.connect()

                queue = await self.channel.declare_queue(
                    self.queue_name,
                    durable=True
                )

                logger.info(f"👀 Starting consumer on queue: {self.queue_name}")

                # Use queue.iterator() instead of queue.consume()
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        try:
                            logger.info(f"📨 Message received! Processing...")
                            await callback(message)
                            logger.info(f"✅ Message processed successfully")
                        except Exception as e:
                            logger.error(
                                f"⚠️ There was an issue processing the message. Here's what happened: {e}", exc_info=True)
                            # Message will be nacked automatically due to exception

            except asyncio.CancelledError:
                logger.info("Consumer task cancelled")
                raise
            except Exception as e:
                logger.error(
                    f"❌ Consumer crashed: {e}, retrying in 5s...", exc_info=True)
                if self.should_reconnect:
                    await asyncio.sleep(5)
                else:
                    break
