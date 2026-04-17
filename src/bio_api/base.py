"""Base class for domain-specific APIs."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import inspect


class BaseAPI(ABC):
    """Base class for domain-specific APIs.

    Provides:
    - Automatic method discovery via list_methods()
    - Help text generation from docstrings
    - Context configuration (workspace paths, etc.)

    Subclasses must implement:
    - api_name property: Return the canonical name of this API
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize API with optional context.

        Args:
            context: Dictionary of configuration values (workspace paths, etc.)
        """
        self._context = context.copy() if context else {}

    def configure(self, **kwargs) -> 'BaseAPI':
        """Configure API context (workspace paths, etc).

        Args:
            **kwargs: Configuration key-value pairs

        Returns:
            Self for method chaining
        """
        self._context.update(kwargs)
        return self

    def list_methods(self) -> List[str]:
        """List all public methods of this API.

        Returns:
            Sorted list of method names (excludes internal methods)
        """
        excluded = {'list_methods', 'help', 'configure', 'api_name'}
        methods = []
        for name in dir(self):
            if name.startswith('_'):
                continue
            if name in excluded:
                continue
            attr = getattr(self, name)
            if callable(attr):
                methods.append(name)
        return sorted(methods)

    def help(self, method_name: Optional[str] = None) -> str:
        """Get help text for API or specific method.

        Args:
            method_name: If provided, get help for this method.
                        If None, get help for the entire API.

        Returns:
            Help text with docstrings and signatures
        """
        if method_name is None:
            # Return API-level help
            doc = self.__class__.__doc__ or f"{self.__class__.__name__} API"
            methods = self.list_methods()
            return f"{doc}\n\nAvailable methods: {methods}"

        method = getattr(self, method_name, None)
        if method is None:
            available = self.list_methods()
            return f"Method '{method_name}' not found. Available: {available}"

        doc = inspect.getdoc(method) or "No documentation available"
        try:
            sig = str(inspect.signature(method))
        except (ValueError, TypeError):
            sig = "(...)"

        return f"{method_name}{sig}\n\n{doc}"

    @property
    @abstractmethod
    def api_name(self) -> str:
        """Return the canonical name of this API."""
        pass
