"""
Ingestion pipeline router
"""
import traceback
import sys
import json
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.schemas.ingestion import IngestionRequest, IngestionResponse
from app.services.service_manager import get_gdrive_service, get_mongodb_service
from app.services.gdrive_service import GoogleDriveService
from app.services.mongodb_service import MongoDBService
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

@lru_cache()
def get_ingestion_service() -> IngestionService:
    """Get or create ingestion service instance (cached)"""
    try:
        service = IngestionService()
        return service
    except Exception as e:
        logger.error(f"Failed to initialize ingestion service: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize ingestion service: {str(e)}"
        )

@router.post("/pipeline")
async def run_ingestion_pipeline(
    payload: IngestionRequest,
    request: Request,
    gdrive_service: GoogleDriveService = Depends(get_gdrive_service),
    mongodb_service: MongoDBService = Depends(get_mongodb_service),
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """
    Run end-to-end ingestion pipeline
    """
    request_id = id(request)
    
    try:
        if not payload.dataset_name:
            raise HTTPException(status_code=400, detail="dataset_name is required")
        if not payload.drive_folder_id:
            raise HTTPException(status_code=400, detail="drive_folder_id is required")
        
        if not gdrive_service:
            raise HTTPException(status_code=503, detail="Google Drive service not initialized")
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="MongoDB service not initialized")
        if not ingestion_service:
            raise HTTPException(status_code=503, detail="Ingestion service not initialized")
        try:
            gdrive_service.list_files_in_folder(
                folder_id=payload.drive_folder_id,
                recursive=True
            )
            
        except Exception as e:
            error_msg = f"Failed to access Google Drive folder: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        result = await ingestion_service.run_pipeline(
            dataset_name=payload.dataset_name,
            drive_folder_id=payload.drive_folder_id,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            force=payload.force,
            metadata_keys=payload.metadata_keys,
            mongodb_service=mongodb_service,
            gdrive_service=gdrive_service
        )
        
        return IngestionResponse(**result)
        
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.status_code} - {e.detail}")
        raise
        
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        logger.error(f"Pipeline failed: {e}")
        
        # Return error response
        return IngestionResponse(
            status="error",
            dataset_name=payload.dataset_name,
            files_processed=0,
            files_embedded=0,
            files_skipped=0,
            embeddings_stored=0,
            metadata_extracted=False,
            pinecone_index=ingestion_service.pinecone_index_name,
            job_id=None,
            message=f"Pipeline failed: {str(e)}",
            error=str(e)
        )