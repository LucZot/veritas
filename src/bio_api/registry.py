"""Central registry for all veritas APIs."""

from typing import Any, Dict, List, Type
from bio_api.base import BaseAPI


class APIRegistry:
    """Central registry for all veritas APIs.

    The registry provides:
    - Discovery: list_apis(), get_api(name)
    - Lazy loading: APIs loaded on first access
    - Global configuration: configure all APIs at once
    - Help system: Help on any API or method

    Usage:
        from bio_api import registry

        # Discovery
        print(registry.list_apis())           # ['sat', 'utils']
        print(registry.sat.list_methods())    # ['load_result', 'calculate_volume', 'calculate_mass', ...]

        # Help
        print(registry.help('sat'))                  # SAT API overview
        print(registry.help('sat.load_result'))      # Method documentation

        # Usage
        result = registry.sat.load_result(db_path, patient_id)
    """

    def __init__(self):
        self._apis: Dict[str, BaseAPI] = {}
        self._api_classes: Dict[str, Type[BaseAPI]] = {}
        self._context: Dict[str, Any] = {}

    def _register_builtin_apis(self):
        """Register the built-in domain APIs (called lazily)."""
        if self._api_classes:
            return  # Already registered

        from bio_api.domains.sat import SATAPI
        from bio_api.domains.utils import UtilsAPI

        self.register_api_class('sat', SATAPI)
        self.register_api_class('utils', UtilsAPI)

    def register_api_class(self, name: str, api_class: Type[BaseAPI]):
        """Register an API class for lazy loading.

        Args:
            name: Name to register the API under (e.g., 'sat')
            api_class: The API class (must inherit from BaseAPI)
        """
        self._api_classes[name] = api_class

    def configure(self, **kwargs) -> 'APIRegistry':
        """Configure global context for all APIs.

        Args:
            workspace_dir: Base workspace directory
            results_db: Default results database path
            ...any other context values

        Returns:
            Self for method chaining
        """
        self._context.update(kwargs)
        # Update existing API instances
        for api in self._apis.values():
            api.configure(**kwargs)
        return self

    def list_apis(self) -> List[str]:
        """List all registered API names.

        Returns:
            Sorted list of available API names
        """
        self._register_builtin_apis()
        return sorted(self._api_classes.keys())

    def get_api(self, name: str) -> BaseAPI:
        """Get an API by name (lazy loading).

        Args:
            name: Name of the API to get

        Returns:
            The API instance

        Raises:
            ValueError: If API name not found
        """
        self._register_builtin_apis()

        if name not in self._apis:
            if name not in self._api_classes:
                available = self.list_apis()
                raise ValueError(
                    f"API '{name}' not found. Available APIs: {available}"
                )
            self._apis[name] = self._api_classes[name](self._context)
        return self._apis[name]

    def help(self, api_or_method: str) -> str:
        """Get help for an API or API method.

        Args:
            api_or_method: API name ('sat') or method ('sat.load_result')

        Returns:
            Help text with docstrings and signatures
        """
        if '.' in api_or_method:
            api_name, method_name = api_or_method.split('.', 1)
            api = self.get_api(api_name)
            return api.help(method_name)
        else:
            api = self.get_api(api_or_method)
            return api.help()

    def __getattr__(self, name: str) -> BaseAPI:
        """Allow attribute access to APIs (registry.sat).

        Args:
            name: API name

        Returns:
            The API instance
        """
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self.get_api(name)
        except ValueError as e:
            raise AttributeError(str(e)) from e

    def __repr__(self) -> str:
        self._register_builtin_apis()
        apis = self.list_apis()
        return f"APIRegistry(apis={apis})"


# Singleton instance
registry = APIRegistry()
