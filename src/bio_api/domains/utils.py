"""Utility functions for data serialization and conversion."""

from typing import Any

from bio_api.base import BaseAPI


class UtilsAPI(BaseAPI):
    """Utility functions for data serialization and conversion.

    Provides helper functions for converting numpy/pandas types to
    JSON-serializable Python types.

    Quick Start:
        from bio_api import registry

        # Convert numpy/pandas data to JSON-safe types
        import numpy as np
        data = {'mean': np.float64(42.5), 'counts': np.array([1, 2, 3])}
        serializable = registry.utils.make_serializable(data)
        json.dumps(serializable)  # Now works without errors

    Methods:
        make_serializable(obj) - Convert numpy/pandas to JSON-safe types
    """

    @property
    def api_name(self) -> str:
        return "utils"

    def make_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas types to JSON-serializable Python types.

        Handles:
        - numpy scalars (np.float64, np.int32, etc.) -> float/int/bool
        - numpy arrays -> lists
        - pandas Series -> lists
        - pandas DataFrames -> dict of lists
        - nested structures (dicts, lists, tuples)

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object

        Example:
            >>> import numpy as np
            >>> data = {'mean': np.float64(42.5), 'counts': np.array([1, 2, 3])}
            >>> serializable = registry.utils.make_serializable(data)
            >>> json.dumps(serializable)  # Now works without errors
        """
        # Handle None
        if obj is None:
            return None

        # Check for numpy types
        try:
            import numpy as np

            # Numpy scalar types
            if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                               np.int16, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.str_):
                return str(obj)

            # Numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass  # numpy not available

        # Check for pandas types
        try:
            import pandas as pd

            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='list')
            if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
        except ImportError:
            pass  # pandas not available

        # Handle nested structures
        if isinstance(obj, dict):
            return {key: self.make_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.make_serializable(item) for item in obj]

        # Primitive types (already serializable)
        if isinstance(obj, (int, float, bool, str)):
            return obj

        # Fallback: try str() conversion
        return str(obj)
