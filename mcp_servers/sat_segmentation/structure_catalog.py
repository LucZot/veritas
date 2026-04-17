"""
Comprehensive catalog of 100+ anatomical structures supported by SAT.

This catalog is extracted from SAT's nii_loader.py which defines loaders for 30+ medical imaging datasets.
Structures are organized by organ system to help agents discover and select appropriate structures for segmentation.

Usage:
    from structure_catalog import STRUCTURE_CATALOG, get_structures_by_category, search_structures

    # Get all cardiac structures
    cardiac = get_structures_by_category("cardiac")

    # Search for kidney-related structures
    kidney_structures = search_structures("kidney")
"""

from typing import List, Dict, Optional

# Comprehensive structure catalog organized by organ system
STRUCTURE_CATALOG = {
    "cardiac": {
        "description": "Heart and cardiovascular structures",
        "structures": [
            "myocardium",
            "left heart ventricle",
            "right heart ventricle",
            "left heart atrium",
            "right heart atrium",
            "heart ventricle",  # combined left + right ventricles
            "heart atrium",  # combined left + right atria
            "heart",
            "heart ascending aorta",
            "pulmonary artery",
        ],
        "datasets": ["ACDC", "MM_WHS_CT", "MM_WHS_MRI", "SegTHOR"]
    },
    "abdominal": {
        "description": "Abdominal organs and structures",
        "structures": [
            "liver",
            "liver tumor",
            "liver cyst",
            "spleen",
            "kidney",
            "left kidney",
            "right kidney",
            "pancreas",
            "stomach",
            "gallbladder",
            "esophagus",
            "duodenum",
            "colon",
            "intestine",
            "small bowel",
            "urinary bladder",
            "rectum",
            "uterus",
        ],
        "datasets": ["BTCV", "AMOS22", "FLARE22", "WORD", "AbdomenCT1K", "CHAOS_MRI", "CHAOS_CT"]
    },
    "vascular": {
        "description": "Blood vessels and vascular structures",
        "structures": [
            "aorta",
            "inferior vena cava",
            "portal vein",
            "splenic vein",
            "portal vein and splenic vein",
            "pulmonary artery",
            "venous system",
            "vena cava",
            "artery",
            "biliary system",
            "hepatic vessel",
        ],
        "datasets": ["BTCV", "AMOS22", "WORD", "SEGA", "PARSE2022", "MSD_HepaticVessel"]
    },
    "spine": {
        "description": "Spine, vertebrae, and intervertebral structures",
        "structures": [
            # Cervical vertebrae (C1-C7)
            "cervical vertebrae 1 (c1)",
            "cervical vertebrae 2 (c2)",
            "cervical vertebrae 3 (c3)",
            "cervical vertebrae 4 (c4)",
            "cervical vertebrae 5 (c5)",
            "cervical vertebrae 6 (c6)",
            "cervical vertebrae 7 (c7)",
            "cervical vertebrae",  # grouped
            # Thoracic vertebrae (T1-T12)
            "thoracic vertebrae 1 (t1)",
            "thoracic vertebrae 2 (t2)",
            "thoracic vertebrae 3 (t3)",
            "thoracic vertebrae 4 (t4)",
            "thoracic vertebrae 5 (t5)",
            "thoracic vertebrae 6 (t6)",
            "thoracic vertebrae 7 (t7)",
            "thoracic vertebrae 8 (t8)",
            "thoracic vertebrae 9 (t9)",
            "thoracic vertebrae 10 (t10)",
            "thoracic vertebrae 11 (t11)",
            "thoracic vertebrae 12 (t12)",
            "thoracic vertebrae",  # grouped
            # Lumbar vertebrae (L1-L6)
            "lumbar vertebrae 1 (l1)",
            "lumbar vertebrae 2 (l2)",
            "lumbar vertebrae 3 (l3)",
            "lumbar vertebrae 4 (l4)",
            "lumbar vertebrae 5 (l5)",
            "lumbar vertebrae 6 (l6)",
            "lumbar vertebrae",  # grouped
            # General
            "vertebrae",  # all vertebrae combined
            "intervertebral discs",
        ],
        "datasets": ["VerSe", "MRSpineSeg"]
    },
    "brain": {
        "description": "Brain and neurological structures",
        "structures": [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            "brain tumor",  # combined tumor regions
        ],
        "datasets": ["BraTS2023_GLI", "BraTS2023_MEN", "BraTS2023_MET", "BraTS2023_PED", "BraTS2023_SSA"]
    },
    "urogenital": {
        "description": "Urogenital and reproductive system structures",
        "structures": [
            "prostate",
            "kidney",
            "left kidney",
            "right kidney",
            "kidney tumor",
            "kidney cyst",
            "urinary bladder",
            "uterus",
            "rectum",
        ],
        "datasets": ["MSD_Prostate", "KiTS23", "BTCV_Cervix"]
    },
    "endocrine": {
        "description": "Endocrine system structures",
        "structures": [
            "adrenal gland",
            "left adrenal gland",
            "right adrenal gland",
            "left adrenal gland tumor",
            "right adrenal gland tumor",
            "adrenal gland tumor",
        ],
        "datasets": ["AMOS22", "FLARE22", "WORD", "IRCADB3D"]
    },
    "musculoskeletal": {
        "description": "Bones and musculoskeletal structures",
        "structures": [
            "bone",
            "head of left femur",
            "head of right femur",
            "head of femur",  # combined
        ],
        "datasets": ["WORD", "IRCADB3D"]
    },
    "respiratory": {
        "description": "Respiratory system structures",
        "structures": [
            "trachea",
            "trachea and bronchie",
            "lung",
            "left lung",
            "right lung",
        ],
        "datasets": ["ATM22", "SegTHOR", "IRCADB3D"]
    },
    "liver_segments": {
        "description": "Detailed liver segmentation (Couinaud classification)",
        "structures": [
            "caudate lobe",
            "left lateral superior segment of liver",
            "left lateral inferior segment of liver",
            "left medial segment of liver",
            "right anterior inferior segment of liver",
            "right posterior inferior segment of liver",
            "right posterior superior segment of liver",
            "right anterior superior segment of liver",
            "left lobe of liver",
            "right lobe of liver",
            "liver",
        ],
        "datasets": ["Couinaud_Liver", "MSD_Liver", "LiQA"]
    }
}

# Flat list of all structures for quick searching
ALL_STRUCTURES = []
for category_data in STRUCTURE_CATALOG.values():
    ALL_STRUCTURES.extend(category_data["structures"])

# Remove duplicates while preserving order
ALL_STRUCTURES = list(dict.fromkeys(ALL_STRUCTURES))


def get_structures_by_category(category: str) -> List[str]:
    """Get all structures for a specific organ system category.

    Args:
        category: Category name (e.g., "cardiac", "abdominal", "spine")

    Returns:
        List of structure names in that category

    Example:
        >>> cardiac_structures = get_structures_by_category("cardiac")
        >>> print(cardiac_structures[:3])
        ['myocardium', 'left heart ventricle', 'right heart ventricle']
    """
    if category not in STRUCTURE_CATALOG:
        available = ", ".join(STRUCTURE_CATALOG.keys())
        raise ValueError(f"Unknown category '{category}'. Available: {available}")

    return STRUCTURE_CATALOG[category]["structures"]


def search_structures(query: str, case_sensitive: bool = False) -> List[str]:
    """Search for structures containing the query string.

    Args:
        query: Search term (e.g., "kidney", "ventricle", "tumor")
        case_sensitive: Whether to perform case-sensitive search

    Returns:
        List of matching structure names

    Example:
        >>> kidney_structures = search_structures("kidney")
        >>> print(kidney_structures)
        ['kidney', 'left kidney', 'right kidney', 'kidney tumor', 'kidney cyst']
    """
    if not case_sensitive:
        query = query.lower()

    matches = []
    for structure in ALL_STRUCTURES:
        search_text = structure if case_sensitive else structure.lower()
        if query in search_text:
            matches.append(structure)

    return matches


def get_category_info(category: str) -> Dict:
    """Get full information about a category including structures and datasets.

    Args:
        category: Category name

    Returns:
        Dictionary with description, structures, and datasets

    Example:
        >>> cardiac_info = get_category_info("cardiac")
        >>> print(cardiac_info["description"])
        'Heart and cardiovascular structures'
    """
    if category not in STRUCTURE_CATALOG:
        available = ", ".join(STRUCTURE_CATALOG.keys())
        raise ValueError(f"Unknown category '{category}'. Available: {available}")

    return STRUCTURE_CATALOG[category]


def get_all_categories() -> List[str]:
    """Get list of all available categories.

    Returns:
        List of category names

    Example:
        >>> categories = get_all_categories()
        >>> print(categories)
        ['cardiac', 'abdominal', 'vascular', 'spine', 'brain', ...]
    """
    return list(STRUCTURE_CATALOG.keys())


def format_catalog_for_agents() -> str:
    """Format the structure catalog as a human-readable string for LLM agents.

    Returns:
        Formatted string describing all available structures

    Example:
        >>> catalog_text = format_catalog_for_agents()
        >>> print(catalog_text[:200])
        'SAT Anatomical Structure Catalog (100+ structures)

        CARDIAC (10 structures): Heart and cardiovascular structures
        - myocardium, left heart ventricle, ...'
    """
    lines = [f"SAT Anatomical Structure Catalog ({len(ALL_STRUCTURES)} structures)\n"]

    for category, data in STRUCTURE_CATALOG.items():
        lines.append(f"\n{category.upper()} ({len(data['structures'])} structures): {data['description']}")
        # Show first 5 structures as examples
        examples = data['structures'][:5]
        examples_str = ", ".join(examples)
        if len(data['structures']) > 5:
            examples_str += f", ... ({len(data['structures']) - 5} more)"
        lines.append(f"  Examples: {examples_str}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print(format_catalog_for_agents())
    print("\n" + "=" * 80 + "\n")

    # Search examples
    print("Search for 'kidney' structures:")
    print(search_structures("kidney"))
    print()

    print("Search for 'ventricle' structures:")
    print(search_structures("ventricle"))
    print()

    print("Get all cardiac structures:")
    print(get_structures_by_category("cardiac"))
