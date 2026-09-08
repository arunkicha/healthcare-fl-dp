"""
Checks that all packages needed for baseline.py (and the later federated.py /
dp_federated.py stages) are installed, and prints versions for reference.

Run from repo root (any directory actually -- doesn't touch data):
    python verify_environment.py
"""

import importlib
import sys

REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "flwr": "flwr",
    "opacus": "opacus",
}

print(f"Python: {sys.version}\n")

missing = []
for module_name, pip_name in REQUIRED.items():
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown version")
        print(f"OK  {pip_name:<20} {version}")
    except ImportError:
        print(f"MISSING  {pip_name:<20} -- run: pip install {pip_name}")
        missing.append(pip_name)

print()
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU -- fine for this dataset size (99 patients), no GPU needed.")
except ImportError:
    pass

if missing:
    print(f"\nInstall missing packages:\n    pip install {' '.join(missing)}")
else:
    print("\nAll required packages are installed.")