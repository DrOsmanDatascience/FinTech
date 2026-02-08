print("Testing imports...")

try:
    print("1. Testing config import...")
    from config import FEATURE_DISPLAY_NAMES
    print(f"   ✓ Config imported. Features: {len(FEATURE_DISPLAY_NAMES)}")
except Exception as e:
    print(f"   ✗ Config import failed: {e}")
    exit(1)

try:
    print("2. Testing visualizations import...")
    from visualizations import create_pca_scatter_plot
    print("   ✓ Visualizations imported successfully!")
except Exception as e:
    print(f"   ✗ Visualizations import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ All imports work!")