"""
Data Versioning System

Manages dataset snapshots for ML model training.
Features:
- Dataset snapshot creation with SHA256 hash
- GCS storage integration
- Snapshot metadata tracking
- MLflow integration for tracking
- Dataset lineage and provenance
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from google.cloud import storage
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..database import get_db
from ..models import DatasetSnapshot, WastePrediction, PantryItem


# ============================================================================
# Configuration
# ============================================================================

GCS_BUCKET_NAME = "pantrypal-datasets"
GCS_DATASETS_PREFIX = "training/"
DATASET_FORMAT = "parquet"  # Efficient columnar format


# ============================================================================
# Models
# ============================================================================

class DatasetMetadata(BaseModel):
    """Metadata for dataset snapshot."""
    total_rows: int
    total_households: int
    date_range_start: datetime
    date_range_end: datetime
    feature_columns: List[str]
    label_columns: List[str]
    data_quality_score: float
    ground_truth_percentage: float
    class_distribution: Dict[str, int]


class SnapshotInfo(BaseModel):
    """Dataset snapshot information."""
    id: int
    hash: str
    gcs_uri: str
    created_at: datetime
    metadata: DatasetMetadata
    size_bytes: int
    format: str


# ============================================================================
# Hash Calculation
# ============================================================================

def calculate_dataset_hash(df: pd.DataFrame) -> str:
    """
    Calculate SHA256 hash of dataset for versioning.
    
    Args:
        df: DataFrame to hash
    
    Returns:
        SHA256 hash as hex string
    """
    # Convert DataFrame to bytes
    # Use deterministic serialization
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    # Calculate SHA256
    hash_obj = hashlib.sha256()
    hash_obj.update(csv_bytes)
    
    return hash_obj.hexdigest()


# ============================================================================
# Dataset Creation
# ============================================================================

def create_training_dataset(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    ground_truth_only: bool = True,
    min_confidence: float = 0.0
) -> pd.DataFrame:
    """
    Create training dataset from waste predictions.
    
    Args:
        db: Database session
        start_date: Filter predictions after this date
        end_date: Filter predictions before this date
        ground_truth_only: Include only verified ground truth labels
        min_confidence: Minimum confidence score for predictions
    
    Returns:
        DataFrame with features and labels
    """
    
    # Query predictions with pantry items
    query = db.query(
        WastePrediction,
        PantryItem
    ).join(
        PantryItem,
        PantryItem.id == WastePrediction.pantry_item_id
    )
    
    # Apply filters
    if start_date:
        query = query.filter(WastePrediction.created_at >= start_date)
    
    if end_date:
        query = query.filter(WastePrediction.created_at <= end_date)
    
    if ground_truth_only:
        query = query.filter(WastePrediction.is_ground_truth == True)
    
    if min_confidence > 0:
        query = query.filter(WastePrediction.confidence_score >= min_confidence)
    
    results = query.all()
    
    # Build dataset
    records = []
    for prediction, pantry_item in results:
        record = {
            # Features
            'product_name': pantry_item.name,
            'category': pantry_item.category,
            'quantity': float(pantry_item.quantity),
            'unit': pantry_item.unit,
            'days_until_expiration': (pantry_item.expiration_date - datetime.utcnow()).days if pantry_item.expiration_date else None,
            'purchase_date': pantry_item.purchase_date,
            'household_size': pantry_item.household.size if pantry_item.household else None,
            'storage_location': pantry_item.storage_location,
            'purchase_price': float(pantry_item.purchase_price) if pantry_item.purchase_price else None,
            
            # Consumption history features
            'historical_consumption_rate': prediction.historical_consumption_rate,
            'days_in_pantry': (datetime.utcnow() - pantry_item.purchase_date).days if pantry_item.purchase_date else None,
            
            # Labels
            'waste_risk_score': prediction.waste_risk_score,
            'days_until_waste': prediction.days_until_waste,
            'waste_reason': prediction.waste_reason,
            
            # Metadata
            'prediction_id': prediction.id,
            'pantry_item_id': pantry_item.id,
            'household_id': pantry_item.household_id,
            'is_ground_truth': prediction.is_ground_truth,
            'confidence_score': prediction.confidence_score,
            'created_at': prediction.created_at,
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    return df


# ============================================================================
# Data Quality Checks
# ============================================================================

def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """
    Calculate data quality score (0-1).
    
    Checks:
    - Missing values percentage
    - Outliers percentage
    - Duplicate percentage
    - Consistency checks
    
    Args:
        df: Dataset DataFrame
    
    Returns:
        Quality score between 0 and 1
    """
    
    if len(df) == 0:
        return 0.0
    
    scores = []
    
    # 1. Missing values (lower is better)
    missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
    missing_score = 1.0 - missing_percentage
    scores.append(missing_score)
    
    # 2. Duplicate rows (lower is better)
    duplicate_percentage = df.duplicated().sum() / len(df)
    duplicate_score = 1.0 - duplicate_percentage
    scores.append(duplicate_score)
    
    # 3. Valid quantity ranges (0 < quantity < 1000)
    if 'quantity' in df.columns:
        valid_quantity = ((df['quantity'] > 0) & (df['quantity'] < 1000)).sum() / len(df)
        scores.append(valid_quantity)
    
    # 4. Valid waste risk scores (0 <= score <= 1)
    if 'waste_risk_score' in df.columns:
        valid_risk = ((df['waste_risk_score'] >= 0) & (df['waste_risk_score'] <= 1)).sum() / len(df)
        scores.append(valid_risk)
    
    # 5. Valid waste reasons (from known taxonomy)
    if 'waste_reason' in df.columns:
        valid_reasons = ['spoilage', 'overcooked', 'portion', 'packaging', 'other']
        valid_reason_percentage = df['waste_reason'].isin(valid_reasons).sum() / len(df)
        scores.append(valid_reason_percentage)
    
    # Average all scores
    overall_score = sum(scores) / len(scores)
    
    return overall_score


def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate dataset and return issues found.
    
    Args:
        df: Dataset DataFrame
    
    Returns:
        Dictionary with validation results
    """
    
    issues = []
    
    # Check for required columns
    required_columns = [
        'product_name', 'category', 'quantity', 'waste_risk_score',
        'days_until_waste', 'waste_reason'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataset
    if len(df) == 0:
        issues.append("Dataset is empty")
    
    # Check for extreme outliers
    if 'quantity' in df.columns:
        outliers = ((df['quantity'] < 0) | (df['quantity'] > 1000)).sum()
        if outliers > 0:
            issues.append(f"Found {outliers} quantity outliers")
    
    # Check for invalid risk scores
    if 'waste_risk_score' in df.columns:
        invalid_scores = ((df['waste_risk_score'] < 0) | (df['waste_risk_score'] > 1)).sum()
        if invalid_scores > 0:
            issues.append(f"Found {invalid_scores} invalid risk scores")
    
    # Check for invalid waste reasons
    if 'waste_reason' in df.columns:
        valid_reasons = ['spoilage', 'overcooked', 'portion', 'packaging', 'other']
        invalid_reasons = ~df['waste_reason'].isin(valid_reasons + [None])
        if invalid_reasons.sum() > 0:
            issues.append(f"Found {invalid_reasons.sum()} invalid waste reasons")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'quality_score': calculate_data_quality_score(df),
    }


# ============================================================================
# GCS Upload
# ============================================================================

def upload_dataset_to_gcs(
    df: pd.DataFrame,
    hash_str: str,
    bucket_name: str = GCS_BUCKET_NAME
) -> str:
    """
    Upload dataset to Google Cloud Storage.
    
    Args:
        df: Dataset DataFrame
        hash_str: SHA256 hash of dataset
        bucket_name: GCS bucket name
    
    Returns:
        GCS URI (gs://bucket/path/to/file)
    """
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Generate filename with timestamp and hash
    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    filename = f"{timestamp}-{hash_str[:8]}.{DATASET_FORMAT}"
    blob_path = f"{GCS_DATASETS_PREFIX}{filename}"
    
    # Upload to GCS
    blob = bucket.blob(blob_path)
    
    # Convert to parquet (efficient columnar format)
    parquet_bytes = df.to_parquet(index=False)
    blob.upload_from_string(parquet_bytes, content_type='application/octet-stream')
    
    # Set metadata
    blob.metadata = {
        'hash': hash_str,
        'created_at': datetime.utcnow().isoformat(),
        'format': DATASET_FORMAT,
        'rows': str(len(df)),
        'columns': str(len(df.columns)),
    }
    blob.patch()
    
    gcs_uri = f"gs://{bucket_name}/{blob_path}"
    
    return gcs_uri


# ============================================================================
# Snapshot Creation
# ============================================================================

def create_dataset_snapshot(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    ground_truth_only: bool = True,
    min_confidence: float = 0.7
) -> SnapshotInfo:
    """
    Create and store a new dataset snapshot.
    
    Args:
        db: Database session
        start_date: Filter start date
        end_date: Filter end date
        ground_truth_only: Include only verified labels
        min_confidence: Minimum confidence threshold
    
    Returns:
        Snapshot information
    """
    
    # Create dataset
    df = create_training_dataset(
        db=db,
        start_date=start_date,
        end_date=end_date,
        ground_truth_only=ground_truth_only,
        min_confidence=min_confidence
    )
    
    if len(df) == 0:
        raise ValueError("No data available for snapshot")
    
    # Validate dataset
    validation = validate_dataset(df)
    if not validation['is_valid']:
        raise ValueError(f"Dataset validation failed: {validation['issues']}")
    
    # Calculate hash
    hash_str = calculate_dataset_hash(df)
    
    # Check if snapshot already exists
    existing = db.query(DatasetSnapshot).filter(
        DatasetSnapshot.hash == hash_str
    ).first()
    
    if existing:
        return SnapshotInfo(
            id=existing.id,
            hash=existing.hash,
            gcs_uri=existing.gcs_uri,
            created_at=existing.created_at,
            metadata=existing.metadata,
            size_bytes=existing.size_bytes,
            format=existing.format,
        )
    
    # Upload to GCS
    gcs_uri = upload_dataset_to_gcs(df, hash_str)
    
    # Calculate metadata
    metadata = DatasetMetadata(
        total_rows=len(df),
        total_households=df['household_id'].nunique(),
        date_range_start=df['created_at'].min(),
        date_range_end=df['created_at'].max(),
        feature_columns=df.columns.tolist(),
        label_columns=['waste_risk_score', 'days_until_waste', 'waste_reason'],
        data_quality_score=validation['quality_score'],
        ground_truth_percentage=df['is_ground_truth'].mean() if 'is_ground_truth' in df.columns else 0.0,
        class_distribution=df['waste_reason'].value_counts().to_dict() if 'waste_reason' in df.columns else {},
    )
    
    # Get file size
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob_path = gcs_uri.replace(f"gs://{GCS_BUCKET_NAME}/", "")
    blob = bucket.blob(blob_path)
    size_bytes = blob.size
    
    # Save snapshot to database
    snapshot = DatasetSnapshot(
        hash=hash_str,
        gcs_uri=gcs_uri,
        created_at=datetime.utcnow(),
        metadata=metadata.dict(),
        size_bytes=size_bytes,
        format=DATASET_FORMAT,
    )
    
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    
    return SnapshotInfo(
        id=snapshot.id,
        hash=snapshot.hash,
        gcs_uri=snapshot.gcs_uri,
        created_at=snapshot.created_at,
        metadata=metadata,
        size_bytes=snapshot.size_bytes,
        format=snapshot.format,
    )


# ============================================================================
# MLflow Integration
# ============================================================================

def log_dataset_to_mlflow(snapshot: SnapshotInfo, run_id: Optional[str] = None):
    """
    Log dataset snapshot to MLflow for experiment tracking.
    
    Args:
        snapshot: Dataset snapshot info
        run_id: MLflow run ID (uses active run if None)
    """
    
    import mlflow
    
    # Log dataset parameters
    mlflow.log_param("dataset_hash", snapshot.hash)
    mlflow.log_param("dataset_uri", snapshot.gcs_uri)
    mlflow.log_param("dataset_size_mb", snapshot.size_bytes / (1024 * 1024))
    mlflow.log_param("dataset_rows", snapshot.metadata.total_rows)
    mlflow.log_param("dataset_households", snapshot.metadata.total_households)
    
    # Log dataset metrics
    mlflow.log_metric("dataset_quality_score", snapshot.metadata.data_quality_score)
    mlflow.log_metric("ground_truth_percentage", snapshot.metadata.ground_truth_percentage)
    
    # Log metadata as artifact
    metadata_json = json.dumps(snapshot.metadata.dict(), indent=2, default=str)
    mlflow.log_text(metadata_json, "dataset_metadata.json")
    
    print(f"Logged dataset snapshot {snapshot.hash[:8]} to MLflow")


# ============================================================================
# Snapshot Retrieval
# ============================================================================

def get_snapshot_by_hash(db: Session, hash_str: str) -> Optional[SnapshotInfo]:
    """
    Get dataset snapshot by hash.
    
    Args:
        db: Database session
        hash_str: SHA256 hash
    
    Returns:
        Snapshot info or None
    """
    
    snapshot = db.query(DatasetSnapshot).filter(
        DatasetSnapshot.hash == hash_str
    ).first()
    
    if not snapshot:
        return None
    
    return SnapshotInfo(
        id=snapshot.id,
        hash=snapshot.hash,
        gcs_uri=snapshot.gcs_uri,
        created_at=snapshot.created_at,
        metadata=DatasetMetadata(**snapshot.metadata),
        size_bytes=snapshot.size_bytes,
        format=snapshot.format,
    )


def list_snapshots(db: Session, limit: int = 10) -> List[SnapshotInfo]:
    """
    List recent dataset snapshots.
    
    Args:
        db: Database session
        limit: Max snapshots to return
    
    Returns:
        List of snapshot info
    """
    
    snapshots = db.query(DatasetSnapshot).order_by(
        DatasetSnapshot.created_at.desc()
    ).limit(limit).all()
    
    return [
        SnapshotInfo(
            id=s.id,
            hash=s.hash,
            gcs_uri=s.gcs_uri,
            created_at=s.created_at,
            metadata=DatasetMetadata(**s.metadata),
            size_bytes=s.size_bytes,
            format=s.format,
        )
        for s in snapshots
    ]
