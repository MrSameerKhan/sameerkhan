"""Smoke test for the sameerkhan Conda environment."""

import sys

import langchain
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers


def main() -> None:
    numpy_values = np.array([1, 2, 3])
    torch_values = torch.tensor([1.0, 2.0, 3.0])

    print("Environment smoke test")
    print(f"python: {sys.version.split()[0]}")
    print(f"numpy: {np.__version__}, sum={numpy_values.sum()}")
    print(f"pandas: {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"torch: {torch.__version__}, sum={float(torch_values.sum())}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"transformers: {transformers.__version__}")
    print(f"langchain: {langchain.__version__}")
    print("OK")


if __name__ == "__main__":
    main()
