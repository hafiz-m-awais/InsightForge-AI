"""
Custom exceptions for the Autonomous Data Science Agent

This module defines custom exception classes to provide more specific
error handling throughout the application.
"""
from typing import Optional, Dict, Any


class DSAgentError(Exception):
    """Base exception class for all DS Agent related errors"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DatasetError(DSAgentError):
    """Raised when there are issues with dataset loading or processing"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATASET_ERROR", details)
        self.file_path = file_path


class ValidationError(DSAgentError):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field


class ModelError(DSAgentError):
    """Raised when there are issues with model training or evaluation"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR", details)
        self.model_name = model_name


class ProcessingError(DSAgentError):
    """Raised when data processing operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PROCESSING_ERROR", details)
        self.operation = operation


class APIError(DSAgentError):
    """Raised when external API calls fail"""
    
    def __init__(self, message: str, service: Optional[str] = None, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "API_ERROR", details)
        self.service = service
        self.status_code = status_code


class ConfigurationError(DSAgentError):
    """Raised when there are configuration issues"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key