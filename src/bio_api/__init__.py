"""bio_api - Discoverable APIs for veritas coding agents.

Usage:
    from bio_api import registry

    # Discovery
    registry.list_apis()              # List available APIs
    registry.sat.list_methods()       # List methods in SAT API
    registry.help('sat.load_structure_mask')  # Get method documentation

    # Usage
    masks = registry.sat.load_structure_mask(db_path, patient_id, 'liver')
    volume = registry.sat.calculate_volume(masks[0]['mask'], masks[0]['spacing'])
    mass = registry.sat.calculate_mass(masks[0]['mask'], masks[0]['spacing'])
"""

from bio_api.registry import registry

__all__ = ['registry']
