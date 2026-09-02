from typing import Optional

from app.services.gdrive_service import GoogleDriveService
from app.services.mongodb_service import MongoDBService

gdrive_service: Optional[GoogleDriveService] = None
mongodb_service: Optional[MongoDBService] = None


def get_gdrive_service() -> GoogleDriveService:
    """
    Get the Google Drive service instance.
    """

    global gdrive_service
    if gdrive_service is None:
        raise RuntimeError("Google Drive service is not available")

    return gdrive_service


def set_gdrive_service(service: GoogleDriveService) -> None:
    """
    Set the Google Drive service instance.
    """
    global gdrive_service
    gdrive_service = service

def get_mongodb_service() -> MongoDBService:
    """
    Get the MongoDB service instance.
    """

    global mongodb_service

    if mongodb_service is None:
        raise RuntimeError("MongoDB service is not available")

    return mongodb_service


def set_mongodb_service(service: MongoDBService) -> None:
    global mongodb_service
    mongodb_service = service
