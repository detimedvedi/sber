"""
Error Handler Module for СберИндекс Anomaly Detection System

This module provides enhanced error handling with:
- Full stack traces with context information
- Data shape and detector name in errors
- Sensitive information sanitization
- Structured error logging
"""

import logging
import traceback
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd


logger = logging.getLogger(__name__)


class SensitiveDataSanitizer:
    """
    Sanitizes sensitive information from error messages and logs.
    
    Removes or masks:
    - File paths (keeps only relative paths from workspace)
    - Personal identifiable information (emails, phone numbers)
    - API keys and tokens
    - Database connection strings
    """
    
    # Patterns for sensitive data
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'api_key': r'\b[A-Za-z0-9]{32,}\b',
        'token': r'(token|key|secret|password)\s*[:=]\s*[\'"]?([^\s\'"]+)[\'"]?',
        'connection_string': r'(mongodb|mysql|postgresql|redis)://[^\s]+',
    }
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize sanitizer.
        
        Args:
            workspace_root: Root directory of workspace for path sanitization
        """
        self.workspace_root = workspace_root or Path.cwd()
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize sensitive information from text.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text with sensitive data masked
        """
        if not text:
            return text
        
        sanitized = text
        
        # Sanitize emails
        sanitized = re.sub(self.PATTERNS['email'], '[EMAIL]', sanitized)
        
        # Sanitize phone numbers
        sanitized = re.sub(self.PATTERNS['phone'], '[PHONE]', sanitized)
        
        # Sanitize API keys (long alphanumeric strings)
        sanitized = re.sub(self.PATTERNS['api_key'], '[API_KEY]', sanitized)
        
        # Sanitize tokens and secrets
        sanitized = re.sub(
            self.PATTERNS['token'], 
            r'\1: [REDACTED]', 
            sanitized, 
            flags=re.IGNORECASE
        )
        
        # Sanitize connection strings
        sanitized = re.sub(self.PATTERNS['connection_string'], r'\1://[REDACTED]', sanitized)
        
        # Sanitize absolute file paths - convert to relative
        sanitized = self._sanitize_paths(sanitized)
        
        return sanitized
    
    def _sanitize_paths(self, text: str) -> str:
        """
        Convert absolute file paths to relative paths.
        
        Args:
            text: Text containing file paths
            
        Returns:
            Text with sanitized paths
        """
        try:
            # Try to convert absolute paths to relative
            workspace_str = str(self.workspace_root.resolve())
            
            # Replace Windows-style paths
            text = text.replace(workspace_str, '.')
            text = text.replace(workspace_str.replace('\\', '/'), '.')
            
            # Replace common user directory patterns
            text = re.sub(r'C:\\Users\\[^\\]+', '[USER_DIR]', text)
            text = re.sub(r'/home/[^/]+', '[USER_DIR]', text)
            text = re.sub(r'/Users/[^/]+', '[USER_DIR]', text)
            
        except Exception as e:
            logger.debug(f"Error sanitizing paths: {e}")
        
        return text
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize sensitive information from dictionary.
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            sanitized_key = self.sanitize_text(str(key))
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[sanitized_key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[sanitized_key] = self.sanitize_dict(value)
            elif isinstance(value, (list, tuple)):
                sanitized[sanitized_key] = [
                    self.sanitize_text(str(item)) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[sanitized_key] = value
        
        return sanitized


class EnhancedErrorContext:
    """
    Captures and formats enhanced error context information.
    
    Includes:
    - Full stack trace
    - Data shape and statistics
    - Detector/component name
    - Configuration context
    - Execution environment
    """
    
    def __init__(self, sanitizer: Optional[SensitiveDataSanitizer] = None):
        """
        Initialize error context handler.
        
        Args:
            sanitizer: Optional sanitizer for sensitive data
        """
        self.sanitizer = sanitizer or SensitiveDataSanitizer()
    
    def capture_context(
        self,
        exception: Exception,
        component_name: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture comprehensive error context.
        
        Args:
            exception: The exception that occurred
            component_name: Name of component/detector where error occurred
            data: DataFrame being processed when error occurred
            config: Configuration dictionary
            additional_context: Additional context information
            
        Returns:
            Dictionary containing all error context
        """
        context = {
            'error_type': type(exception).__name__,
            'error_message': str(exception),
            'component': component_name or 'unknown',
        }
        
        # Add stack trace
        context['stack_trace'] = self._format_stack_trace(exception)
        
        # Add data context if available
        if data is not None:
            context['data_context'] = self._capture_data_context(data)
        
        # Add configuration context (sanitized)
        if config is not None:
            context['config_context'] = self._capture_config_context(config)
        
        # Add additional context
        if additional_context:
            context['additional_context'] = self.sanitizer.sanitize_dict(additional_context)
        
        # Sanitize the entire context
        context = self.sanitizer.sanitize_dict(context)
        
        return context
    
    def _format_stack_trace(self, exception: Exception) -> List[str]:
        """
        Format full stack trace with context.
        
        Args:
            exception: The exception to format
            
        Returns:
            List of stack trace lines
        """
        # Get full traceback
        tb_lines = traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__
        )
        
        # Sanitize each line
        sanitized_lines = [
            self.sanitizer.sanitize_text(line.rstrip())
            for line in tb_lines
        ]
        
        return sanitized_lines
    
    def _capture_data_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Capture data shape and statistics.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with data context
        """
        context = {
            'shape': data.shape,
            'rows': len(data),
            'columns': len(data.columns),
            'column_names': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
        }
        
        # Add memory usage
        try:
            memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            context['memory_usage_mb'] = round(memory_mb, 2)
        except Exception:
            context['memory_usage_mb'] = 'unknown'
        
        # Add missing value statistics
        try:
            missing_counts = data.isnull().sum()
            if missing_counts.sum() > 0:
                context['missing_values'] = {
                    col: int(count)
                    for col, count in missing_counts.items()
                    if count > 0
                }
                context['missing_percentage'] = round(
                    (missing_counts.sum() / (len(data) * len(data.columns))) * 100,
                    2
                )
        except Exception:
            pass
        
        # Add numeric column statistics
        try:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                context['numeric_columns_count'] = len(numeric_cols)
                
                # Sample statistics for first few numeric columns
                sample_stats = {}
                for col in list(numeric_cols)[:5]:  # Limit to first 5
                    try:
                        sample_stats[col] = {
                            'min': float(data[col].min()),
                            'max': float(data[col].max()),
                            'mean': float(data[col].mean()),
                            'null_count': int(data[col].isnull().sum())
                        }
                    except Exception:
                        pass
                
                if sample_stats:
                    context['sample_statistics'] = sample_stats
        except Exception:
            pass
        
        return context
    
    def _capture_config_context(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Capture relevant configuration context.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Sanitized configuration context
        """
        # Extract only relevant config sections
        relevant_keys = [
            'detection_profile',
            'thresholds',
            'temporal',
            'municipality_classification',
            'robust_statistics',
            'export'
        ]
        
        context = {}
        for key in relevant_keys:
            if key in config:
                context[key] = config[key]
        
        return context
    
    def format_error_message(self, context: Dict[str, Any]) -> str:
        """
        Format comprehensive error message for logging.
        
        Args:
            context: Error context dictionary
            
        Returns:
            Formatted error message
        """
        lines = [
            "=" * 80,
            f"ERROR IN COMPONENT: {context.get('component', 'unknown')}",
            "=" * 80,
            "",
            f"Error Type: {context.get('error_type', 'Unknown')}",
            f"Error Message: {context.get('error_message', 'No message')}",
            ""
        ]
        
        # Add data context
        if 'data_context' in context:
            data_ctx = context['data_context']
            lines.extend([
                "Data Context:",
                f"  Shape: {data_ctx.get('shape', 'unknown')}",
                f"  Rows: {data_ctx.get('rows', 'unknown')}",
                f"  Columns: {data_ctx.get('columns', 'unknown')}",
            ])
            
            if 'memory_usage_mb' in data_ctx:
                lines.append(f"  Memory Usage: {data_ctx['memory_usage_mb']} MB")
            
            if 'missing_percentage' in data_ctx:
                lines.append(f"  Missing Values: {data_ctx['missing_percentage']}%")
            
            lines.append("")
        
        # Add configuration context
        if 'config_context' in context:
            lines.extend([
                "Configuration Context:",
                f"  Profile: {context['config_context'].get('detection_profile', 'unknown')}",
                ""
            ])
        
        # Add additional context
        if 'additional_context' in context:
            lines.append("Additional Context:")
            for key, value in context['additional_context'].items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Add stack trace
        if 'stack_trace' in context:
            lines.extend([
                "Stack Trace:",
                *context['stack_trace'],
                ""
            ])
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


class ErrorHandler:
    """
    Main error handler for the anomaly detection system.
    
    Provides methods for handling errors with enhanced context,
    logging, and sanitization.
    """
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize error handler.
        
        Args:
            workspace_root: Root directory of workspace
        """
        self.sanitizer = SensitiveDataSanitizer(workspace_root)
        self.context_handler = EnhancedErrorContext(self.sanitizer)
        self.logger = logging.getLogger(__name__)
    
    def handle_error(
        self,
        exception: Exception,
        component_name: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        log_level: int = logging.ERROR
    ) -> Dict[str, Any]:
        """
        Handle an error with full context capture and logging.
        
        Args:
            exception: The exception that occurred
            component_name: Name of component where error occurred
            data: DataFrame being processed
            config: Configuration dictionary
            additional_context: Additional context information
            log_level: Logging level (default: ERROR)
            
        Returns:
            Error context dictionary
        """
        # Capture full context
        context = self.context_handler.capture_context(
            exception=exception,
            component_name=component_name,
            data=data,
            config=config,
            additional_context=additional_context
        )
        
        # Format error message
        error_message = self.context_handler.format_error_message(context)
        
        # Log the error
        self.logger.log(
            log_level,
            error_message,
            extra={
                'component': component_name,
                'error_type': context.get('error_type'),
                'data_shape': context.get('data_context', {}).get('shape'),
            }
        )
        
        return context
    
    def log_warning_with_context(
        self,
        message: str,
        component_name: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """
        Log a warning with context information.
        
        Args:
            message: Warning message
            component_name: Name of component
            data: DataFrame being processed
            additional_context: Additional context
        """
        # Sanitize message
        sanitized_message = self.sanitizer.sanitize_text(message)
        
        # Build extra context
        extra = {'component': component_name}
        
        if data is not None:
            extra['data_shape'] = data.shape
            extra['data_rows'] = len(data)
            extra['data_columns'] = len(data.columns)
        
        if additional_context:
            extra.update(self.sanitizer.sanitize_dict(additional_context))
        
        # Log warning
        self.logger.warning(sanitized_message, extra=extra)


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler(workspace_root: Optional[Path] = None) -> ErrorHandler:
    """
    Get or create global error handler instance.
    
    Args:
        workspace_root: Root directory of workspace
        
    Returns:
        ErrorHandler instance
    """
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(workspace_root)
    
    return _global_error_handler
