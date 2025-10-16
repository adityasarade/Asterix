"""
Unit tests for ParameterConstraint validation system.

Tests all validation types supported by ParameterConstraint:
- Min/max value validation (numbers)
- Min/max length validation (strings, lists)
- Pattern validation (regex)
- Allowed values validation (enum-like)
- Custom validator functions
- Edge cases and error messages
"""

import pytest
import re
from asterix.tools.base import ParameterConstraint


class TestMinMaxValueValidation:
    """Test min_value and max_value constraints for numbers."""
    
    def test_min_value_integer_valid(self):
        """Test integer passes when >= min_value."""
        constraint = ParameterConstraint(min_value=10)
        is_valid, error = constraint.validate(15, "age")
        
        assert is_valid is True
        assert error is None
    
    def test_min_value_integer_invalid(self):
        """Test integer fails when < min_value."""
        constraint = ParameterConstraint(min_value=10)
        is_valid, error = constraint.validate(5, "age")
        
        assert is_valid is False
        assert "age must be >= 10" in error
        assert "got 5" in error
    
    def test_max_value_integer_valid(self):
        """Test integer passes when <= max_value."""
        constraint = ParameterConstraint(max_value=100)
        is_valid, error = constraint.validate(75, "score")
        
        assert is_valid is True
        assert error is None
    
    def test_max_value_integer_invalid(self):
        """Test integer fails when > max_value."""
        constraint = ParameterConstraint(max_value=100)
        is_valid, error = constraint.validate(150, "score")
        
        assert is_valid is False
        assert "score must be <= 100" in error
        assert "got 150" in error
    
    def test_min_max_value_range_valid(self):
        """Test value passes when within min/max range."""
        constraint = ParameterConstraint(min_value=1, max_value=10)
        is_valid, error = constraint.validate(5, "rating")
        
        assert is_valid is True
        assert error is None
    
    def test_min_max_value_below_range(self):
        """Test value fails when below minimum."""
        constraint = ParameterConstraint(min_value=1, max_value=10)
        is_valid, error = constraint.validate(0, "rating")
        
        assert is_valid is False
        assert "rating must be >= 1" in error
    
    def test_min_max_value_above_range(self):
        """Test value fails when above maximum."""
        constraint = ParameterConstraint(min_value=1, max_value=10)
        is_valid, error = constraint.validate(11, "rating")
        
        assert is_valid is False
        assert "rating must be <= 10" in error
    
    def test_min_value_float_valid(self):
        """Test float value validation."""
        constraint = ParameterConstraint(min_value=0.5)
        is_valid, error = constraint.validate(1.75, "threshold")
        
        assert is_valid is True
        assert error is None
    
    def test_max_value_float_invalid(self):
        """Test float fails when exceeding max."""
        constraint = ParameterConstraint(max_value=1.0)
        is_valid, error = constraint.validate(1.5, "probability")
        
        assert is_valid is False
        assert "probability must be <= 1.0" in error


class TestMinMaxLengthValidation:
    """Test min_length and max_length constraints for strings and lists."""
    
    def test_min_length_string_valid(self):
        """Test string passes when length >= min_length."""
        constraint = ParameterConstraint(min_length=3)
        is_valid, error = constraint.validate("hello", "username")
        
        assert is_valid is True
        assert error is None
    
    def test_min_length_string_invalid(self):
        """Test string fails when length < min_length."""
        constraint = ParameterConstraint(min_length=5)
        is_valid, error = constraint.validate("hi", "username")
        
        assert is_valid is False
        assert "username length must be >= 5" in error
        assert "got 2" in error
    
    def test_max_length_string_valid(self):
        """Test string passes when length <= max_length."""
        constraint = ParameterConstraint(max_length=10)
        is_valid, error = constraint.validate("short", "name")
        
        assert is_valid is True
        assert error is None
    
    def test_max_length_string_invalid(self):
        """Test string fails when length > max_length."""
        constraint = ParameterConstraint(max_length=5)
        is_valid, error = constraint.validate("verylongstring", "code")
        
        assert is_valid is False
        assert "code length must be <= 5" in error
        assert "got 14" in error
    
    def test_min_length_list_valid(self):
        """Test list passes when length >= min_length."""
        constraint = ParameterConstraint(min_length=2)
        is_valid, error = constraint.validate([1, 2, 3], "items")
        
        assert is_valid is True
        assert error is None
    
    def test_max_length_list_invalid(self):
        """Test list fails when length > max_length."""
        constraint = ParameterConstraint(max_length=3)
        is_valid, error = constraint.validate([1, 2, 3, 4, 5], "tags")
        
        assert is_valid is False
        assert "tags length must be <= 3" in error
        assert "got 5" in error
    
    def test_empty_string_with_min_length(self):
        """Test empty string fails min_length validation."""
        constraint = ParameterConstraint(min_length=1)
        is_valid, error = constraint.validate("", "field")
        
        assert is_valid is False
        assert "field length must be >= 1" in error


class TestPatternValidation:
    """Test pattern (regex) validation for strings."""
    
    def test_pattern_alphanumeric_valid(self):
        """Test alphanumeric pattern matches valid input."""
        constraint = ParameterConstraint(pattern=r'^[a-zA-Z0-9]+$')
        is_valid, error = constraint.validate("user123", "username")
        
        assert is_valid is True
        assert error is None
    
    def test_pattern_alphanumeric_invalid(self):
        """Test alphanumeric pattern rejects special characters."""
        constraint = ParameterConstraint(pattern=r'^[a-zA-Z0-9]+$')
        is_valid, error = constraint.validate("user@123", "username")
        
        assert is_valid is False
        assert "username must match pattern" in error
    
    def test_pattern_email_valid(self):
        """Test email pattern validation."""
        constraint = ParameterConstraint(pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        is_valid, error = constraint.validate("user@example.com", "email")
        
        assert is_valid is True
        assert error is None
    
    def test_pattern_email_invalid(self):
        """Test email pattern rejects invalid format."""
        constraint = ParameterConstraint(pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        is_valid, error = constraint.validate("not-an-email", "email")
        
        assert is_valid is False
        assert "email must match pattern" in error


class TestAllowedValuesValidation:
    """Test allowed_values (enum-like) validation."""
    
    def test_allowed_values_string_valid(self):
        """Test string in allowed values passes."""
        constraint = ParameterConstraint(allowed_values=["red", "green", "blue"])
        is_valid, error = constraint.validate("red", "color")
        
        assert is_valid is True
        assert error is None
    
    def test_allowed_values_string_invalid(self):
        """Test string not in allowed values fails."""
        constraint = ParameterConstraint(allowed_values=["red", "green", "blue"])
        is_valid, error = constraint.validate("yellow", "color")
        
        assert is_valid is False
        assert "color must be one of" in error
        assert "got yellow" in error
    
    def test_allowed_values_integer_valid(self):
        """Test integer in allowed values passes."""
        constraint = ParameterConstraint(allowed_values=[1, 2, 3, 5, 8])
        is_valid, error = constraint.validate(5, "fibonacci")
        
        assert is_valid is True
        assert error is None
    
    def test_allowed_values_integer_invalid(self):
        """Test integer not in allowed values fails."""
        constraint = ParameterConstraint(allowed_values=[1, 2, 3, 5, 8])
        is_valid, error = constraint.validate(4, "fibonacci")
        
        assert is_valid is False
        assert "fibonacci must be one of" in error


class TestCustomValidatorValidation:
    """Test custom_validator function validation."""
    
    def test_custom_validator_valid(self):
        """Test custom validator passes when returns True."""
        def is_even(x):
            return x % 2 == 0
        
        constraint = ParameterConstraint(custom_validator=is_even)
        is_valid, error = constraint.validate(4, "number")
        
        assert is_valid is True
        assert error is None
    
    def test_custom_validator_invalid(self):
        """Test custom validator fails when returns False."""
        def is_even(x):
            return x % 2 == 0
        
        constraint = ParameterConstraint(custom_validator=is_even)
        is_valid, error = constraint.validate(5, "number")
        
        assert is_valid is False
        assert "number failed custom validation" in error
    
    def test_custom_validator_with_exception(self):
        """Test custom validator handles exceptions gracefully."""
        def buggy_validator(x):
            raise ValueError("Something went wrong")
        
        constraint = ParameterConstraint(custom_validator=buggy_validator)
        is_valid, error = constraint.validate("test", "param")
        
        assert is_valid is False
        assert "validation error" in error.lower()
        assert "Something went wrong" in error


class TestCombinedConstraints:
    """Test multiple constraints working together."""
    
    def test_length_and_pattern_valid(self):
        """Test value passes both length and pattern constraints."""
        constraint = ParameterConstraint(
            min_length=3,
            max_length=20,
            pattern=r'^[a-zA-Z0-9_]+$'
        )
        is_valid, error = constraint.validate("user123", "username")
        
        assert is_valid is True
        assert error is None
    
    def test_length_fails_pattern_valid(self):
        """Test fails on length even if pattern matches."""
        constraint = ParameterConstraint(
            min_length=5,
            pattern=r'^[a-zA-Z0-9_]+$'
        )
        is_valid, error = constraint.validate("ab", "username")
        
        assert is_valid is False
        assert "length must be >= 5" in error
    
    def test_pattern_fails_length_valid(self):
        """Test fails on pattern even if length valid."""
        constraint = ParameterConstraint(
            min_length=3,
            pattern=r'^[a-zA-Z]+$'  # Only letters
        )
        is_valid, error = constraint.validate("user123", "username")
        
        assert is_valid is False
        assert "must match pattern" in error