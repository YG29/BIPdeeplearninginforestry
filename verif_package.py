import importlib

def verify_required_packages():
    """
    Check if all required packages are installed in the environment.

    Returns:
        None. Prints status of each package.
    """
    required_packages = {
        'skimage': 'skimage',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'os': 'os',  # built-in, no check needed
        'joblib': 'joblib'
    }

    print("🔍 Verifying required packages...\n")

    for import_name, pip_name in required_packages.items():
        if import_name == 'os':
            print(f"✅ {import_name} (built-in)")
            continue
        try:
            importlib.import_module(import_name)
            print(f"✅ {import_name}")
        except ImportError:
            print(f"❌ {import_name} is NOT installed. Install it using:")
            print(f"   pip install {pip_name}\n")


# Example usage:
if __name__ == "__main__":
    verify_required_packages()

