#!/usr/bin/env python3
"""
Test that all analysis modules can be imported and basic functionality works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test all analysis module imports."""
    print("Testing analysis module imports...")
    print("=" * 60)

    try:
        print("\n1. Calibration module...")
        from analysis.calibration import (
            compute_expected_calibration_error,
            compute_compositional_calibration,
        )
        print("   ✓ calibration imports successful")
    except Exception as e:
        print(f"   ✗ calibration import failed: {e}")
        return False

    try:
        print("\n2. Similarity module...")
        from analysis.similarity import (
            compute_cka,
            analyze_representation_similarity,
        )
        print("   ✓ similarity imports successful")
    except Exception as e:
        print(f"   ✗ similarity import failed: {e}")
        return False

    try:
        print("\n3. Geometry module...")
        from analysis.geometry import (
            compute_effective_dimensionality,
            analyze_representation_geometry,
        )
        print("   ✓ geometry imports successful")
    except Exception as e:
        print(f"   ✗ geometry import failed: {e}")
        return False

    try:
        print("\n4. Statistics module...")
        from analysis.statistics import (
            bootstrap_ci,
            paired_t_test,
            compute_cohens_d,
        )
        print("   ✓ statistics imports successful")
    except Exception as e:
        print(f"   ✗ statistics import failed: {e}")
        return False

    try:
        print("\n5. Probes module...")
        from analysis.probes import (
            DigitIdentityProbe,
            StrokeStructureProbe,
            train_stroke_structure_probe,
        )
        print("   ✓ probes imports successful")
    except Exception as e:
        print(f"   ✗ probes import failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
