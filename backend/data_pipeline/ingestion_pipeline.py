"""
Data Ingestion Pipeline - Message Queue Processing

Handles ingestion from multiple sources:
- Manual client input (API)
- Barcode scans
- Receipt OCR
- Retailer APIs (OAuth2)
- Smart fridge/IoT sensors (MQTT)

Architecture:
- Pub/Sub message queue (Google Cloud Pub/Sub or Kafka)
- Worker processes for normalization & canonicalization
- Background prediction generation
- Monitoring & metrics
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import asyncio
from uuid import uuid4

from google.cloud import pubsub_v1  # or use kafka-python
import aioredis

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert, select

from ..shared.models_v2 import (
    PantryEntry,
    Purchase,
    ItemsCatalog,
    IngestionLog,
)
from ..ml.services.canonicalization_service import ItemCanonicalizationService
from ..ml.models.embedding_matcher import EmbeddingMatcher
from .unit_normalizer import UnitNormalizer
from .geo_enrichment import GeoEnrichmentService


logger = logging.getLogger(__name__)


class IngestionSource(str, Enum):
    """Data source types."""
    MANUAL_INPUT = "manual_input"
    BARCODE_SCAN = "barcode_scan"
    RECEIPT_OCR = "receipt_ocr"
    RETAILER_API = "retailer_api"
    IOT_SENSOR = "iot_sensor"
    SMART_FRIDGE = "smart_fridge"


class IngestionStatus(str, Enum):
    """Ingestion processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    CANONICALIZED = "canonicalized"
    NEEDS_USER_INPUT = "needs_user_input"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionMessage:
    """Raw ingestion message format."""
    message_id: str
    source: IngestionSource
    household_id: str
    user_id: str
    timestamp: datetime
    
    # Item data
    item_name: Optional[str] = None
    barcode: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = "USD"
    
    # Receipt OCR data
    receipt_image_url: Optional[str] = None
    raw_ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    
    # Retailer API data
    retailer_id: Optional[str] = None
    external_product_id: Optional[str] = None
    
    # IoT sensor data
    sensor_id: Optional[str] = None
    weight_grams: Optional[float] = None
    temperature_celsius: Optional[float] = None
    
    # Purchase metadata
    purchase_date: Optional[datetime] = None
    purchase_location: Optional[str] = None
    expiry_date: Optional[datetime] = None
    storage_location: Optional[str] = None
    
    # Enrichment data
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionMessage":
        """Create from dictionary."""
        # Convert ISO strings to datetime
        for key in ["timestamp", "purchase_date", "expiry_date"]:
            if key in data and data[key]:
                data[key] = datetime.fromisoformat(data[key])
        
        # Convert enum strings
        if "source" in data:
            data["source"] = IngestionSource(data["source"])
        
        return cls(**data)


@dataclass
class IngestionResult:
    """Result of ingestion processing."""
    message_id: str
    status: IngestionStatus
    
    # Canonicalization results
    canonical_item_id: Optional[str] = None
    match_confidence: Optional[float] = None
    candidate_items: Optional[List[Dict[str, Any]]] = None
    
    # Created records
    pantry_entry_id: Optional[str] = None
    purchase_id: Optional[str] = None
    
    # Normalization
    normalized_quantity: Optional[float] = None
    normalized_unit: Optional[str] = None
    
    # Error handling
    error: Optional[str] = None
    retry_count: int = 0
    
    # Processing metadata
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None


class IngestionPipeline:
    """Main ingestion pipeline coordinator."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        pubsub_project: str,
        pubsub_topic: str,
        pubsub_subscription: str,
        redis_url: str = "redis://localhost:6379",
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            db_session: Database session
            pubsub_project: Google Cloud project ID
            pubsub_topic: Pub/Sub topic name
            pubsub_subscription: Pub/Sub subscription name
            redis_url: Redis connection URL
        """
        self.db = db_session
        
        # Pub/Sub client
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        self.topic_path = self.publisher.topic_path(pubsub_project, pubsub_topic)
        self.subscription_path = self.subscriber.subscription_path(
            pubsub_project, pubsub_subscription
        )
        
        # Redis for deduplication & caching
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        
        # Services
        self.unit_normalizer = UnitNormalizer()
        self.geo_enrichment = GeoEnrichmentService()
        self.canonicalization_service: Optional[ItemCanonicalizationService] = None
        
        logger.info("IngestionPipeline initialized")
    
    async def initialize(self):
        """Initialize async dependencies."""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
        # Initialize embedding matcher (load FAISS index)
        embedding_matcher = EmbeddingMatcher()
        embedding_matcher.load_index("models/faiss_index")
        
        self.canonicalization_service = ItemCanonicalizationService(
            self.db,
            embedding_matcher,
        )
        
        logger.info("Async dependencies initialized")
    
    def publish_message(self, message: IngestionMessage) -> str:
        """
        Publish message to ingestion queue.
        
        Args:
            message: IngestionMessage to publish
            
        Returns:
            Message ID
        """
        # Serialize message
        message_data = json.dumps(message.to_dict()).encode("utf-8")
        
        # Publish to Pub/Sub
        future = self.publisher.publish(
            self.topic_path,
            data=message_data,
            source=message.source.value,
            household_id=message.household_id,
        )
        
        message_id = future.result()
        
        logger.info(
            f"Published message {message.message_id} to {self.topic_path} "
            f"(source: {message.source})"
        )
        
        # Track in Redis for deduplication
        asyncio.create_task(self._track_message(message.message_id))
        
        return message_id
    
    async def _track_message(self, message_id: str):
        """Track message ID in Redis for deduplication."""
        if self.redis:
            await self.redis.setex(
                f"ingestion:seen:{message_id}",
                3600,  # 1 hour TTL
                "1",
            )
    
    async def _is_duplicate(self, message_id: str) -> bool:
        """Check if message was already processed."""
        if self.redis:
            exists = await self.redis.exists(f"ingestion:seen:{message_id}")
            return exists == 1
        return False
    
    async def process_message(self, message: IngestionMessage) -> IngestionResult:
        """
        Process ingestion message.
        
        Pipeline stages:
        1. Deduplication check
        2. Unit normalization
        3. Canonicalization (barcode/fuzzy/embedding)
        4. Create purchase & pantry_entry records
        5. Trigger background prediction job
        6. Log metrics
        
        Args:
            message: IngestionMessage to process
            
        Returns:
            IngestionResult
        """
        start_time = datetime.utcnow()
        
        logger.info(f"Processing message {message.message_id} from {message.source}")
        
        try:
            # Stage 1: Deduplication
            if await self._is_duplicate(message.message_id):
                logger.warning(f"Duplicate message {message.message_id}, skipping")
                return IngestionResult(
                    message_id=message.message_id,
                    status=IngestionStatus.COMPLETED,
                    error="Duplicate message",
                )
            
            # Stage 2: Unit normalization
            normalized_quantity, normalized_unit = self.unit_normalizer.normalize(
                message.quantity or 1.0,
                message.unit or "piece",
            )
            
            logger.debug(
                f"Normalized: {message.quantity} {message.unit} "
                f"→ {normalized_quantity} {normalized_unit}"
            )
            
            # Stage 3: Canonicalization
            canonicalization_result = await self.canonicalization_service.canonicalize(
                user_input=message.item_name or "",
                barcode=message.barcode,
                metadata={
                    "source": message.source.value,
                    "purchase_location": message.purchase_location,
                },
            )
            
            # Stage 4: Create records
            if canonicalization_result.canonical_item_id and not canonicalization_result.needs_human_review:
                # High confidence match - create pantry entry
                pantry_entry_id, purchase_id = await self._create_pantry_entry(
                    message,
                    canonicalization_result.canonical_item_id,
                    normalized_quantity,
                    normalized_unit,
                )
                
                # Stage 5: Trigger prediction job
                await self._trigger_prediction_job(pantry_entry_id)
                
                status = IngestionStatus.COMPLETED
            else:
                # Low confidence or no match - needs user input
                pantry_entry_id = None
                purchase_id = None
                status = IngestionStatus.NEEDS_USER_INPUT
                
                logger.info(
                    f"Item '{message.item_name}' needs user input "
                    f"(confidence: {canonicalization_result.match_confidence:.2f})"
                )
            
            # Stage 6: Save raw OCR data (if receipt)
            if message.source == IngestionSource.RECEIPT_OCR:
                await self._save_receipt_data(message, purchase_id)
            
            # Stage 7: Log ingestion metrics
            processing_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            
            await self._log_ingestion_metrics(
                message,
                status,
                processing_time_ms,
                canonicalization_result.match_confidence,
            )
            
            # Create result
            result = IngestionResult(
                message_id=message.message_id,
                status=status,
                canonical_item_id=canonicalization_result.canonical_item_id,
                match_confidence=canonicalization_result.match_confidence,
                candidate_items=canonicalization_result.candidate_items,
                pantry_entry_id=pantry_entry_id,
                purchase_id=purchase_id,
                normalized_quantity=normalized_quantity,
                normalized_unit=normalized_unit,
                processing_time_ms=processing_time_ms,
                model_version="v1.0.0",  # TODO: Get from model registry
            )
            
            logger.info(
                f"Message {message.message_id} processed successfully "
                f"(status: {status}, time: {processing_time_ms:.0f}ms)"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to process message {message.message_id}: {e}")
            
            return IngestionResult(
                message_id=message.message_id,
                status=IngestionStatus.FAILED,
                error=str(e),
            )
    
    async def _create_pantry_entry(
        self,
        message: IngestionMessage,
        canonical_item_id: str,
        normalized_quantity: float,
        normalized_unit: str,
    ) -> tuple[str, str]:
        """Create purchase and pantry_entry records."""
        purchase_id = str(uuid4())
        pantry_entry_id = str(uuid4())
        
        # Geo enrichment
        enrichment_data = await self.geo_enrichment.enrich(
            location=message.purchase_location,
            purchase_date=message.purchase_date or datetime.utcnow(),
        )
        
        # Create purchase record
        purchase = Purchase(
            purchase_id=purchase_id,
            household_id=message.household_id,
            user_id=message.user_id,
            purchase_date=message.purchase_date or datetime.utcnow(),
            store_name=message.purchase_location,
            total_cost=message.price,
            payment_method="unknown",
            raw_ocr=message.raw_ocr_text,
            ocr_confidence=message.ocr_confidence,
            source=message.source.value,
            metadata=message.metadata,
            created_at=datetime.utcnow(),
        )
        
        self.db.add(purchase)
        
        # Create pantry entry
        pantry_entry = PantryEntry(
            entry_id=pantry_entry_id,
            household_id=message.household_id,
            item_id=canonical_item_id,
            purchase_id=purchase_id,
            quantity=normalized_quantity,
            quantity_on_hand=normalized_quantity,
            unit=normalized_unit,
            purchase_date=message.purchase_date or datetime.utcnow(),
            expiry_date=message.expiry_date,
            storage_location=message.storage_location or "pantry",
            price_per_unit=message.price / normalized_quantity if message.price else None,
            added_by=message.user_id,
            barcode=message.barcode,
            status="active",
            created_at=datetime.utcnow(),
        )
        
        self.db.add(pantry_entry)
        
        await self.db.commit()
        
        logger.info(
            f"Created pantry_entry {pantry_entry_id} and purchase {purchase_id}"
        )
        
        return pantry_entry_id, purchase_id
    
    async def _save_receipt_data(
        self,
        message: IngestionMessage,
        purchase_id: Optional[str],
    ):
        """Save raw receipt data for debugging."""
        # Store receipt image & OCR data
        # NOTE: Retention policy - delete after X days unless user consents
        
        if message.receipt_image_url:
            # Upload to cloud storage (GCS/S3) with lifecycle policy
            # Set expiration: 30 days by default
            pass
        
        if message.raw_ocr_text and purchase_id:
            # Update purchase record with OCR data
            await self.db.execute(
                f"UPDATE purchases SET raw_ocr = :ocr, ocr_confidence = :conf "
                f"WHERE purchase_id = :id",
                {
                    "ocr": message.raw_ocr_text,
                    "conf": message.ocr_confidence,
                    "id": purchase_id,
                },
            )
    
    async def _trigger_prediction_job(self, pantry_entry_id: str):
        """Trigger background job to generate initial prediction."""
        # Publish to prediction queue
        prediction_message = {
            "pantry_entry_id": pantry_entry_id,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": "normal",
        }
        
        # Publish to separate prediction topic
        prediction_topic = self.publisher.topic_path(
            self.topic_path.split("/")[1],  # Extract project
            "waste-prediction-jobs",
        )
        
        future = self.publisher.publish(
            prediction_topic,
            data=json.dumps(prediction_message).encode("utf-8"),
        )
        
        message_id = future.result()
        
        logger.info(
            f"Triggered prediction job for pantry_entry {pantry_entry_id} "
            f"(message: {message_id})"
        )
    
    async def _log_ingestion_metrics(
        self,
        message: IngestionMessage,
        status: IngestionStatus,
        processing_time_ms: float,
        match_confidence: Optional[float],
    ):
        """Log ingestion metrics to monitoring."""
        # Create ingestion log record
        log_entry = IngestionLog(
            log_id=str(uuid4()),
            message_id=message.message_id,
            source=message.source.value,
            household_id=message.household_id,
            user_id=message.user_id,
            status=status.value,
            processing_time_ms=processing_time_ms,
            match_confidence=match_confidence,
            timestamp=datetime.utcnow(),
            metadata={
                "barcode": message.barcode,
                "has_receipt": message.receipt_image_url is not None,
                "retailer_id": message.retailer_id,
                "sensor_id": message.sensor_id,
            },
        )
        
        self.db.add(log_entry)
        await self.db.commit()
        
        # Push metrics to monitoring system (Prometheus)
        # METRICS.ingestion_processed_total.labels(
        #     source=message.source.value,
        #     status=status.value,
        # ).inc()
        # 
        # METRICS.ingestion_processing_time.labels(
        #     source=message.source.value,
        # ).observe(processing_time_ms / 1000)
        
        if match_confidence:
            # METRICS.canonicalization_confidence.labels(
            #     source=message.source.value,
            # ).observe(match_confidence)
            pass
    
    def start_worker(self, num_workers: int = 4):
        """
        Start worker processes to consume messages.
        
        Args:
            num_workers: Number of concurrent workers
        """
        logger.info(f"Starting {num_workers} ingestion workers")
        
        def callback(message: pubsub_v1.subscriber.message.Message):
            """Process Pub/Sub message."""
            try:
                # Deserialize
                ingestion_message = IngestionMessage.from_dict(
                    json.loads(message.data.decode("utf-8"))
                )
                
                # Process
                result = asyncio.run(self.process_message(ingestion_message))
                
                # ACK if successful
                if result.status != IngestionStatus.FAILED:
                    message.ack()
                else:
                    # NACK - message will be redelivered
                    message.nack()
                    logger.error(f"Message {result.message_id} failed: {result.error}")
            
            except Exception as e:
                logger.error(f"Worker error: {e}")
                message.nack()
        
        # Subscribe with flow control
        flow_control = pubsub_v1.types.FlowControl(
            max_messages=num_workers,
            max_bytes=10 * 1024 * 1024,  # 10 MB
        )
        
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=callback,
            flow_control=flow_control,
        )
        
        logger.info(f"Listening for messages on {self.subscription_path}")
        
        # Block until interrupted
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Worker stopped")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    
    async def main():
        # Initialize pipeline
        engine = create_async_engine("postgresql+asyncpg://localhost/pantrypal")
        async with AsyncSession(engine) as session:
            pipeline = IngestionPipeline(
                db_session=session,
                pubsub_project="pantrypal-prod",
                pubsub_topic="ingestion-queue",
                pubsub_subscription="ingestion-worker",
            )
            
            await pipeline.initialize()
            
            # Example: Manual input
            message = IngestionMessage(
                message_id=str(uuid4()),
                source=IngestionSource.MANUAL_INPUT,
                household_id="household-123",
                user_id="user-456",
                timestamp=datetime.utcnow(),
                item_name="Organic Whole Milk",
                barcode="012345678901",
                quantity=1.0,
                unit="gallon",
                price=4.99,
                purchase_date=datetime.utcnow(),
                storage_location="refrigerator",
            )
            
            # Publish to queue
            message_id = pipeline.publish_message(message)
            print(f"Published message: {message_id}")
            
            # Or process directly (for testing)
            result = await pipeline.process_message(message)
            print(f"Result: {result}")
    
    asyncio.run(main())
