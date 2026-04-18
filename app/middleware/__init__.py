"""
Middleware package for the Autonomous Data Science Agent

Provides centralized error handling, logging, and request/response processing.
"""
from .exceptions import (
    DSAgentError, DatasetError, ValidationError, ModelError,
    ProcessingError, APIError, ConfigurationError
)
from .error_handler import (
    ErrorHandler, safe_execute, global_exception_handler,
    validate_required_fields, validate_file_path
)

__all__ = [
    'DSAgentError', 'DatasetError', 'ValidationError', 'ModelError',
    'ProcessingError', 'APIError', 'ConfigurationError',
    'ErrorHandler', 'safe_execute', 'global_exception_handler',
    'validate_required_fields', 'validate_file_path'
]