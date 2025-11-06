"""
Tests for enhanced error handling module.
"""

import pytest
import pandas as pd
import logging
from pathlib import Path
from src.error_handler import (
    SensitiveDataSanitizer,
    EnhancedErrorContext,
    ErrorHandler,
    get_error_handler
)


class TestSensitiveDataSanitizer:
    """Tests for SensitiveDataSanitizer class."""
    
    def test_sanitize_email(self):
        """Test email sanitization."""
        sanitizer = SensitiveDataSanitizer()
        
        text = "Contact user@example.com for details"
        result = sanitizer.sanitize_text(text)
        
        assert "[EMAIL]" in result
        assert "user@example.com" not in result
    
    def test_sanitize_phone(self):
        """Test phone number sanitization."""
        sanitizer = SensitiveDataSanitizer()
        
        text = "Call 123-456-7890 for support"
        result = sanitizer.sanitize_text(text)
        
        assert "[PHONE]" in result
        assert "123-456-7890" not in result
    
    def test_sanitize_api_key(self):
        """Test API key sanitization."""
        sanitizer = SensitiveDataSanitizer()
        
        text = "API key: abc123def456ghi789jkl012mno345pq"
        result = sanitizer.sanitize_text(text)
        
        assert "[API_KEY]" in result or "[REDACTED]" in result
    
    def test_sanitize_paths(self):
        """Test file path sanitization."""
        sanitizer = SensitiveDataSanitizer(workspace_root=Path("/home/user/project"))
        
        text = "Error in /home/user/project/src/data_loader.py"
        result = sanitizer.sanitize_text(text)
        
        assert "/home/user" not in result or "[USER_DIR]" in result
    
    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        sanitizer = SensitiveDataSanitizer()
        
        data = {
            'email': 'test@example.com',
            'phone': '123-456-7890',
            'name': 'John Doe'
        }
        
        result = sanitizer.sanitize_dict(data)
        
        assert "[EMAIL]" in str(result.values())
        assert "[PHONE]" in str(result.values())
        assert "test@example.com" not in str(result.values())


class TestEnhancedErrorContext:
    """Tests for EnhancedErrorContext class."""
    
    def test_capture_basic_context(self):
        """Test basic error context capture."""
        context_handler = EnhancedErrorContext()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            context = context_handler.capture_context(
                exception=e,
                component_name="TestComponent"
            )
        
        assert context['error_type'] == 'ValueError'
        assert context['error_message'] == 'Test error'
        assert context['component'] == 'TestComponent'
        assert 'stack_trace' in context
        assert len(context['stack_trace']) > 0
    
    def test_capture_data_context(self):
        """Test data context capture."""
        context_handler = EnhancedErrorContext()
        
        # Create sample dataframe
        df = pd.DataFrame({
            'col1': [1, 2, 3, None, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            context = context_handler.capture_context(
                exception=e,
                component_name="TestComponent",
                data=df
            )
        
        assert 'data_context' in context
        data_ctx = context['data_context']
        
        # Shape might be list or tuple after sanitization
        assert data_ctx['shape'] == (5, 2) or data_ctx['shape'] == [5, 2]
        assert data_ctx['rows'] == 5
        assert data_ctx['columns'] == 2
        assert 'col1' in data_ctx['column_names']
        assert 'col2' in data_ctx['column_names']
        assert 'missing_values' in data_ctx
    
    def test_capture_config_context(self):
        """Test configuration context capture."""
        context_handler = EnhancedErrorContext()
        
        config = {
            'detection_profile': 'normal',
            'thresholds': {
                'statistical': {'z_score': 3.0}
            },
            'sensitive_key': 'secret_value'
        }
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            context = context_handler.capture_context(
                exception=e,
                component_name="TestComponent",
                config=config
            )
        
        assert 'config_context' in context
        config_ctx = context['config_context']
        
        assert 'detection_profile' in config_ctx
        assert 'thresholds' in config_ctx
    
    def test_format_error_message(self):
        """Test error message formatting."""
        context_handler = EnhancedErrorContext()
        
        context = {
            'component': 'TestComponent',
            'error_type': 'ValueError',
            'error_message': 'Test error',
            'data_context': {
                'shape': (100, 10),
                'rows': 100,
                'columns': 10
            },
            'stack_trace': ['Line 1', 'Line 2']
        }
        
        message = context_handler.format_error_message(context)
        
        assert 'TestComponent' in message
        assert 'ValueError' in message
        assert 'Test error' in message
        assert 'Shape: (100, 10)' in message
        assert 'Stack Trace:' in message


class TestErrorHandler:
    """Tests for ErrorHandler class."""
    
    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = ErrorHandler()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            context = handler.handle_error(
                exception=e,
                component_name="TestComponent"
            )
        
        assert context is not None
        assert context['error_type'] == 'ValueError'
        assert context['component'] == 'TestComponent'
    
    def test_handle_error_with_data(self):
        """Test error handling with data context."""
        handler = ErrorHandler()
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        try:
            raise KeyError("Missing key")
        except Exception as e:
            context = handler.handle_error(
                exception=e,
                component_name="DataProcessor",
                data=df
            )
        
        assert context is not None
        assert 'data_context' in context
        assert context['data_context']['rows'] == 3
    
    def test_log_warning_with_context(self, caplog):
        """Test warning logging with context."""
        handler = ErrorHandler()
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with caplog.at_level(logging.WARNING):
            handler.log_warning_with_context(
                message="Test warning",
                component_name="TestComponent",
                data=df
            )
        
        assert "Test warning" in caplog.text
    
    def test_get_error_handler_singleton(self):
        """Test global error handler singleton."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        assert handler1 is handler2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
