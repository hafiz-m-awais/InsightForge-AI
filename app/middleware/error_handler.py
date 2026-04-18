"""
Error handling utilities for the Autonomous Data Science Agent

This module provides standardized error handling, logging, and response formatting.
"""
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .exceptions import (
    DSAgentError, DatasetError, ValidationError, ModelError, 
    ProcessingError, APIError, ConfigurationError
)

# Configure logging
logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling and response formatting"""
    
    ERROR_CODES = {
        ValidationError: 400,
        DatasetError: 404,
        ModelError: 422,
        ProcessingError: 500,
        APIError: 502,
        ConfigurationError: 500,
        DSAgentError: 500,
    }
    
    @staticmethod
    def format_error_response(error: Exception, request_id: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """Format an exception into a standardized error response"""
        
        if isinstance(error, DSAgentError):
            return ErrorHandler._format_custom_error(error, request_id)
        elif isinstance(error, HTTPException):
            return ErrorHandler._format_http_error(error, request_id)
        else:
            return ErrorHandler._format_generic_error(error, request_id)
    
    @staticmethod
    def _format_custom_error(error: DSAgentError, request_id: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """Format custom DS Agent errors"""
        status_code = ErrorHandler.ERROR_CODES.get(type(error), 500)
        
        response = {
            "success": False,
            "error_code": error.error_code,
            "message": error.message,
            "details": error.details,
        }
        
        if request_id:
            response["request_id"] = request_id
            
        # Add specific error attributes
        if isinstance(error, DatasetError) and error.file_path:
            response["file_path"] = error.file_path
        elif isinstance(error, ModelError) and error.model_name:
            response["model_name"] = error.model_name
        elif isinstance(error, ValidationError) and error.field:
            response["field"] = error.field
            
        return response, status_code
    
    @staticmethod
    def _format_http_error(error: HTTPException, request_id: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """Format FastAPI HTTP exceptions"""
        response = {
            "success": False,
            "error_code": "HTTP_ERROR",
            "message": error.detail,
            "details": {}
        }
        
        if request_id:
            response["request_id"] = request_id
            
        return response, error.status_code
    
    @staticmethod
    def _format_generic_error(error: Exception, request_id: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
        """Format generic Python exceptions"""
        error_message = str(error) if str(error) else "An unexpected error occurred"
        
        response = {
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": error_message,
            "details": {
                "error_type": error.__class__.__name__
            }
        }
        
        if request_id:
            response["request_id"] = request_id
            
        return response, 500


def safe_execute(operation_name: str = None, log_errors: bool = True):
    """Decorator for safe execution of functions with standardized error handling"""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {operation_name or func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={"function": func.__name__, "args": str(args)[:200]}
                    )
                
                response_data, status_code = ErrorHandler.format_error_response(e)
                raise HTTPException(status_code=status_code, detail=response_data)
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {operation_name or func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={"function": func.__name__, "args": str(args)[:200]}
                    )
                
                # For sync functions, we'll re-raise the custom exceptions
                # The calling code should handle conversion to HTTP responses
                raise e
                
        # Return the appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Exception handler for FastAPI
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for FastAPI application"""
    request_id = getattr(request.state, 'request_id', None)
    
    response_data, status_code = ErrorHandler.format_error_response(exc, request_id)
    
    # Log the error
    logger.error(
        f"Unhandled exception in {request.method} {request.url}: {str(exc)}",
        exc_info=True,
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": status_code
        }
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Validate that required fields are present in data"""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            details={"missing_fields": missing_fields}
        )


def validate_file_path(file_path: str, base_directory: str) -> str:
    """Validate that a file path is within the allowed base directory"""
    import os
    from pathlib import Path
    
    try:
        # Resolve the absolute paths
        base_path = Path(base_directory).resolve()
        target_path = Path(file_path).resolve()
        
        # Check if the target path is within the base directory
        if not str(target_path).startswith(str(base_path)):
            raise ValidationError(
                f"File path '{file_path}' is outside allowed directory",
                details={"file_path": file_path, "base_directory": base_directory}
            )
            
        # Check if file exists
        if not target_path.exists():
            raise DatasetError(
                f"File not found: {file_path}",
                file_path=file_path
            )
            
        return str(target_path)
        
    except Exception as e:
        if isinstance(e, (ValidationError, DatasetError)):
            raise
        else:
            raise ValidationError(f"Invalid file path: {str(e)}", details={"file_path": file_path})