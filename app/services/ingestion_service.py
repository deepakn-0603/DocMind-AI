"""
Ingestion service for end-to-end pipeline: OCR, metadata extraction, embedding, and Pinecone storage
"""
import json
import os
import re
import traceback
import sys
from typing import List, Dict, Any, Optional
from loguru import logger

from pinecone import Pinecone, ServerlessSpec

from app.services.service_manager import get_gdrive_service, get_mongodb_service
from app.services.gdrive_service import GoogleDriveService
from app.services.ocr_service import OCRService
from app.services.mongodb_service import MongoDBService
from app.config import settings
from sentence_transformers import SentenceTransformer

class IngestionService:
    """Service for end-to-end ingestion pipeline using LOCAL models"""
    
    def __init__(self):
        """Initialize ingestion service with local models and Pinecone"""
        
        self.local_embedding_model = None
        self.pinecone_client = None
        self.pinecone_index_name = None
        self.pinecone_index = None
        
        try:
            self.local_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Local embedding model loaded")
        except Exception as e:
            self.local_embedding_model = None
        
        # Initialize Pinecone client
        pinecone_api_key = settings.PINECONE_API_KEY
        if not pinecone_api_key:
            error_msg = "PINECONE_API_KEY configuration is required"
            raise ValueError(error_msg)
        
        try:
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        except Exception as e:
            raise
        
        self.pinecone_index_name = settings.PINECONE_INDEX_NAME
        if not self.pinecone_index_name:
            error_msg = "PINECONE_INDEX_NAME configuration is required"
            raise ValueError(error_msg)
        
        embedding_dimension = settings.EMBEDDING_DIMENSION
        if not embedding_dimension:
            error_msg = "EMBEDDING_DIMENSION configuration is required"
            raise ValueError(error_msg)
        
        try:
            existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
            
            if self.pinecone_index_name not in existing_indexes:
                self.pinecone_client.create_index(
                    name=self.pinecone_index_name,
                    dimension=embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
            
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise
        
    def _get_default_metadata_schema(self) -> Dict[str, Any]:
        """Get default legal metadata schema"""
        return {
            "document_id": "string",
            "title": "string",
            "court_name": "string",
            "case_number": "string",
            "case_type": "string",
            "decision_date": "YYYY-MM-DD",
            "coram": "string",
            "petitioner": "string",
            "respondent": "string",
            "key_issues": "string"
        }
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using LOCAL sentence-transformers model
        """
        try:
            if self.local_embedding_model:
                embedding = self.local_embedding_model.encode(text).tolist()
                return embedding
            else:
                return self._fallback_embedding(text)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Hash-based fallback embedding"""
        import hashlib
        import numpy as np
        
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(settings.EMBEDDING_DIMENSION)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    async def extract_metadata(
        self,
        full_text: str,
        metadata_keys: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata using rule-based extraction (always works)
        """
        if metadata_keys is None:
            metadata_keys = self._get_default_metadata_schema()
        
        logger.info("Extracting metadata using rule-based approach")
        return self._extract_metadata_rule_based(full_text, metadata_keys)
    
    def _extract_metadata_rule_based(self, full_text: str, metadata_keys: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based metadata extraction for legal documents"""
        metadata = {}
        
        # Document ID
        doc_id_patterns = [
            r'(?:Doc(?:ument)?\s*(?:ID|No|Number)?[:#]?\s*([A-Z0-9\-/]+))',
            r'(?:C\.O\.|CP|Diary)\s*No\.?\s*([A-Z0-9\-/]+)',
            r'(?:Case|WP|W\.P\.|Crl|Civil|Appeal)\s*(?:No|Number)?\.?\s*[:#]?\s*([A-Z0-9\-/]+(?:\s*of\s*\d{4})?)',
            r'([A-Z]+/\d+/\d{4})',
        ]
        metadata['document_id'] = ""
        for pattern in doc_id_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                metadata['document_id'] = match.group(1) if match.lastindex else match.group(0)
                break
        
        # Title - first meaningful line
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        if lines:
            skip_prefixes = ['IN THE', 'BEFORE', 'CASE NO', 'WP', 'W.P', 'CRL', 'CIVIL']
            for line in lines[:5]:
                if not any(line.upper().startswith(prefix) for prefix in skip_prefixes):
                    metadata['title'] = line[:200]
                    break
            else:
                metadata['title'] = lines[0][:200]
        else:
            metadata['title'] = ""
        
        # Court name
        court_match = re.search(r'(?:IN\s+THE\s+)?([A-Z\s]+COURT[A-Z\s]*)', full_text)
        metadata['court_name'] = court_match.group(1).strip() if court_match else ""
        
        # Case number
        case_match = re.search(r'(?:Case|Crl|Cr\.|WP|W\.P\.|Civil|Appeal)\s*(?:No|Number)?\.?\s*[:#]?\s*([A-Z0-9\-/]+)', full_text, re.IGNORECASE)
        metadata['case_number'] = case_match.group(0) if case_match else ""
        
        # Case type
        metadata['case_type'] = ""
        if metadata['case_number']:
            if re.search(r'WP|W\.P\.', metadata['case_number'], re.IGNORECASE):
                metadata['case_type'] = "Writ Petition"
            elif re.search(r'Crl|Cr\.', metadata['case_number'], re.IGNORECASE):
                metadata['case_type'] = "Criminal"
            elif re.search(r'Civil', metadata['case_number'], re.IGNORECASE):
                metadata['case_type'] = "Civil"
        
        # Date
        date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', full_text)
        metadata['decision_date'] = date_match.group(1) if date_match else ""
        
        # Coram
        coram_match = re.search(r'(?:Coram[:.]?\s*)([^\n]+)', full_text, re.IGNORECASE)
        metadata['coram'] = coram_match.group(1).strip()[:200] if coram_match else ""
        
        # Petitioner
        pet_match = re.search(r'(?:Petitioner[s]?[:.]?\s*)([^\n]+?)(?=\s*(?:Respondent|Vs|Versus|v\.|$))', full_text, re.IGNORECASE)
        metadata['petitioner'] = pet_match.group(1).strip()[:200] if pet_match else ""
        
        # Respondent
        resp_match = re.search(r'(?:Respondent[s]?[:.]?\s*)([^\n]+?)(?=\s*(?:Coram|Judgment|Order|Date|$))', full_text, re.IGNORECASE)
        metadata['respondent'] = resp_match.group(1).strip()[:200] if resp_match else ""
        
        # Key issues
        metadata['key_issues'] = ""
        
        # Fill remaining
        for key in metadata_keys:
            if key not in metadata:
                metadata[key] = ""
        
        return metadata
    
    def _normalize_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metadata for Pinecone"""
        normalized = {}
        for key, value in metadata.items():
            if value is None:
                normalized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) > 1000:
                    normalized[key] = value[:1000]
                else:
                    normalized[key] = value
            elif isinstance(value, list):
                normalized[key] = ", ".join(str(item) for item in value if item)
            elif isinstance(value, dict):
                normalized[key] = json.dumps(value)
            else:
                normalized[key] = str(value)
        return normalized

    async def run_pipeline(
        self,
        dataset_name: str,
        drive_folder_id: str,
        chunk_size: int,
        chunk_overlap: int,
        force: bool = False,
        metadata_keys: Optional[Dict[str, Any]] = None,
        mongodb_service: Optional[MongoDBService] = None,
        gdrive_service: Optional[GoogleDriveService] = None
    ) -> Dict[str, Any]:
        """
        Run end-to-end ingestion pipeline
        
        Args:
            dataset_name: Name of the dataset
            drive_folder_id: Google Drive folder ID
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters
            force: Force reprocessing even if already processed
            metadata_keys: Optional custom metadata schema
            mongodb_service: Optional MongoDB service (will get from service manager if None)
            gdrive_service: Optional Google Drive service (will get from service manager if None)
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Starting ingestion pipeline for dataset: {dataset_name}")
        # Get services
        if gdrive_service is None:
            gdrive_service = get_gdrive_service()
        if mongodb_service is None:
            mongodb_service = get_mongodb_service()
        
        # Resolve root folder (same logic as OCR endpoint)
        root_folder_id = drive_folder_id or settings.GOOGLE_DRIVE_FOLDER_ID
        
        if not root_folder_id:
            raise ValueError("drive_folder_id must be provided or set in environment")
        
        # Ensure output root folder exists (mirror structure)
        output_root_folder_id = gdrive_service.create_or_get_folder(
            folder_name="Optical Character Recognition",
            parent_folder_id=root_folder_id
        )
        
        # Ensure output dataset folder exists
        output_dataset_folder_id = gdrive_service.create_or_get_folder(
            folder_name=dataset_name,
            parent_folder_id=output_root_folder_id
        )
        
        # Create job in MongoDB
        job_id = mongodb_service.create_job(
            dataset_name=dataset_name,
            input_folder_id=drive_folder_id,
            output_folder_id=output_dataset_folder_id
        )
        
        # Initialize OCR service
        ocr_service = OCRService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # List files in folder
        try:
            files = gdrive_service.list_files_in_folder(
                folder_id=drive_folder_id,
                recursive=True
            )
            logger.info(f"Found {len(files)} files to process")
        except Exception as e:
            logger.error(f"Failed to list files in folder '{drive_folder_id}': {e}")
            mongodb_service.finish_job(job_id, "failed")
            error_msg = str(e)
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise ValueError(
                    f"Folder '{drive_folder_id}' not found or inaccessible. "
                    f"Please verify the folder ID and ensure you have access to it."
                )
            raise ValueError(f"Failed to access folder '{drive_folder_id}': {error_msg}")
        
        mongodb_service.update_job_counters(job_id, {"files_discovered": len(files)})
        
        files_processed = 0
        files_skipped = 0
        files_failed = 0
        embeddings_stored = 0
        files_embedded = 0
        files_embedding_failed = 0
        
        # Process each file
        for file_info in files:
            file_id = file_info['id']
            file_name = file_info['name']
            file_path = file_info['path']
            mime_type = file_info.get('mimeType')
            
            logger.info(f"Processing file: {file_name}")
            
            # Construct Google Drive public view URL
            source_url = f"https://drive.google.com/file/d/{file_id}/view"
            
            # Create document record in MongoDB
            doc_id = mongodb_service.create_doc(
                job_id=job_id,
                dataset_name=dataset_name,
                source_drive_file_id=file_id,
                source_path=file_path,
                source_url=source_url
            )
            
            # Determine output path (mirror structure under "Optical Character Recognition")
            # Same logic as OCR endpoint
            if file_path.startswith(f"{dataset_name}/"):
                relative_path = file_path[len(f"{dataset_name}/"):]
            elif file_path == dataset_name:
                relative_path = ""
            else:
                relative_path = file_path
            
            # Get directory and filename
            if relative_path:
                dir_path = os.path.dirname(relative_path)
                base_name = os.path.splitext(file_name)[0]
                output_file_name = f"{base_name}.json"
                
                if dir_path:
                    output_path = f"{dataset_name}/{dir_path}/{output_file_name}"
                    output_dir_path = f"{dataset_name}/{dir_path}"
                else:
                    output_path = f"{dataset_name}/{output_file_name}"
                    output_dir_path = dataset_name
            else:
                base_name = os.path.splitext(file_name)[0]
                output_file_name = f"{base_name}.json"
                output_path = f"{dataset_name}/{output_file_name}"
                output_dir_path = dataset_name
            
            # Ensure output folder hierarchy exists
            output_parent_folder_id = output_dataset_folder_id
            if output_dir_path and output_dir_path != dataset_name:
                subfolder_path = output_dir_path.replace(f"{dataset_name}/", "")
                if subfolder_path:
                    output_parent_folder_id = gdrive_service.ensure_folder_hierarchy(
                        folder_path=subfolder_path,
                        parent_folder_id=output_dataset_folder_id
                    )
            
            # Check if already embedded (unless force)
            skip_embedding = False
            if not force:
                if mongodb_service.is_doc_embedded(file_id, dataset_name):
                    logger.info(f"Skipping embedding for {file_name}: already embedded")
                    skip_embedding = True
            
            # Check if OCR JSON already exists (unless force)
            existing_json_file_id = None
            if not force:
                existing_json_file_id = gdrive_service.file_exists_in_folder(
                    file_name=output_file_name,
                    folder_id=output_parent_folder_id
                )
            
            try:
                mongodb_service.update_doc_status(doc_id, "processing", "Downloading file")
                
                # Download file
                file_bytes = gdrive_service.download_file(file_id)
                
                # Process OCR document
                ocr_result = ocr_service.process_document(
                    file_bytes=file_bytes,
                    file_name=file_name,
                    mime_type=mime_type
                )
                
                # Check if document has no chunks
                if ocr_result['chunks_emitted'] == 0:
                    mongodb_service.update_doc_status(doc_id, "failed", "No extractable text found")
                    mongodb_service.update_doc_counts(doc_id, {
                        "pages_total": ocr_result['total_page_count'],
                        "pages_without_text": ocr_result['pages_without_text'],
                        "chunks_emitted": 0,
                        "lang_undetected_count": 1 if ocr_result['lang_undetected'] else 0
                    })
                    mongodb_service.update_job_counters(job_id, {
                        "files_failed": 1,
                        "pages_processed": ocr_result['total_page_count'],
                        "pages_without_text": ocr_result['pages_without_text'],
                        "lang_undetected_count": 1 if ocr_result['lang_undetected'] else 0
                    })
                    files_failed += 1
                    continue
                
                # Store OCR JSON file (if not exists or force)
                if not existing_json_file_id or force:
                    json_chunks = []
                    for chunk in ocr_result['chunks']:
                        json_chunks.append({
                            "doc_id": chunk['doc_id'],
                            "file_name": chunk['file_name'],
                            "language": chunk['language'],
                            "total_page_count": chunk['total_page_count'],
                            "page_index": chunk['page_index'],
                            "chunk_index": chunk['chunk_index'],
                            "text": chunk['text'],
                            "source_url": source_url
                        })
                    
                    json_content = json.dumps(json_chunks, ensure_ascii=False, indent=2)
                    json_bytes = json_content.encode('utf-8')
                    
                    output_file_info = gdrive_service.upload_file_from_bytes(
                        file_bytes=json_bytes,
                        file_name=output_file_name,
                        folder_id=output_parent_folder_id,
                        mime_type="application/json"
                    )
                    
                    output_drive_file_id = output_file_info['file_id']
                    mongodb_service.update_doc_output(doc_id, output_drive_file_id, output_path)
                else:
                    logger.info(f"OCR JSON already exists for {file_name}, skipping upload")
                    output_drive_file_id = existing_json_file_id
                    mongodb_service.update_doc_output(doc_id, output_drive_file_id, output_path)
                    mongodb_service.update_doc_status(doc_id, "skipped", "OCR JSON already exists")
                
                # Update OCR counts
                mongodb_service.update_doc_counts(doc_id, {
                    "pages_total": ocr_result['total_page_count'],
                    "pages_without_text": ocr_result['pages_without_text'],
                    "chunks_emitted": ocr_result['chunks_emitted'],
                    "lang_undetected_count": 1 if ocr_result['lang_undetected'] else 0
                })
                
                mongodb_service.update_job_counters(job_id, {
                    "files_processed": 1,
                    "pages_processed": ocr_result['total_page_count'],
                    "pages_without_text": ocr_result['pages_without_text'],
                    "chunks_emitted": ocr_result['chunks_emitted'],
                    "lang_undetected_count": 1 if ocr_result['lang_undetected'] else 0
                })
                
                files_processed += 1
                
                # Extract metadata for embedding
                if not skip_embedding:
                    try:
                        mongodb_service.update_doc_status(doc_id, "processing", "Extracting metadata and generating embeddings")
                        
                        # Extract full text for metadata extraction
                        full_text, _, _ = ocr_service.extract_text(
                            file_bytes=file_bytes,
                            file_name=file_name,
                            mime_type=mime_type
                        )
                        
                        # Extract metadata using GPT
                        metadata = await self.extract_metadata(full_text, metadata_keys)
                        
                        # Generate embeddings and upsert to Pinecone
                        vectors_to_upsert = []
                        for chunk in ocr_result['chunks']:
                            chunk_text = chunk['text']
                            chunk_index = chunk['chunk_index']
                            page_index = chunk['page_index']
                            
                            # Generate embedding
                            embedding = self.generate_embedding(chunk_text)
                            
                            # Prepare metadata for Pinecone
                            chunk_metadata = {
                                "dataset_name": dataset_name,
                                "source_file": file_name,
                                "text": chunk_text,
                                "chunk_index": chunk_index,
                                "page_index": page_index,
                                "source_url": source_url
                            }
                            
                            # Add extracted metadata
                            chunk_metadata.update(metadata)
                            
                            # Normalize metadata for Pinecone compatibility
                            chunk_metadata = self._normalize_metadata_for_pinecone(chunk_metadata)
                            
                            # Generate unique ID with page_index for better idempotence
                            doc_id_str = metadata.get("document_id", f"doc_{file_name}_{files_processed}")
                            doc_id_str = str(doc_id_str).replace(" ", "_").replace("/", "_")[:50]
                            vector_id = f"{doc_id_str}_p{page_index}_c{chunk_index}"
                            
                            vectors_to_upsert.append({
                                "id": vector_id,
                                "values": embedding,
                                "metadata": chunk_metadata
                            })
                            
                            embeddings_stored += 1
                        
                        # Upsert to Pinecone
                        if vectors_to_upsert:
                            self.pinecone_index.upsert(vectors=vectors_to_upsert)
                            
                            # Mark as embedded in MongoDB
                            mongodb_service.mark_doc_embedded(
                                doc_id=doc_id,
                                embeddings_count=len(vectors_to_upsert),
                                pinecone_index=self.pinecone_index_name
                            )
                            
                            mongodb_service.update_job_embedding_counters(job_id, {
                                "files_embedded": 1,
                                "embeddings_stored": len(vectors_to_upsert)
                            })
                            
                            files_embedded += 1
                            logger.success(
                                f"Upserted {len(vectors_to_upsert)} vectors to Pinecone for: {file_name}"
                            )
                        
                        mongodb_service.update_doc_status(doc_id, "ok", "Processed and embedded successfully")
                        
                    except Exception as e:
                        logger.error(f"Error embedding file {file_name}: {e}")
                        mongodb_service.update_doc_status(doc_id, "ok", f"OCR completed but embedding failed: {str(e)}")
                        mongodb_service.update_job_embedding_counters(job_id, {"files_embedding_failed": 1})
                        files_embedding_failed += 1
                        if force:
                            raise
                else:
                    mongodb_service.update_doc_status(doc_id, "ok", "OCR completed, embedding skipped (already embedded)")
                    files_skipped += 1
                
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                mongodb_service.update_doc_status(doc_id, "failed", str(e))
                mongodb_service.update_job_counters(job_id, {"files_failed": 1})
                files_failed += 1
                if force:
                    raise
        
        # Determine final job status
        if files_failed == 0:
            job_status = "completed"
        elif files_processed > 0 or files_embedded > 0:
            job_status = "completed_with_errors"
        else:
            job_status = "failed"
        
        mongodb_service.finish_job(job_id, job_status)
        
        logger.success(
            f"Ingestion pipeline completed: {files_processed} files processed, "
            f"{files_embedded} embedded, {embeddings_stored} embeddings stored"
        )
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "files_processed": files_processed,
            "files_embedded": files_embedded,
            "files_skipped": files_skipped,
            "embeddings_stored": embeddings_stored,
            "metadata_extracted": True,
            "pinecone_index": self.pinecone_index_name,
            "job_id": job_id,
            "message": "Ingestion pipeline completed successfully."
        }

