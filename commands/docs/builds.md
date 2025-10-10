| # | Build Type | Output Format | Purpose / When Used | Example Tools |
|---|-------------|---------------|---------------------|----------------|
| **1**  | **Source Distribution**                         | `.tar.gz`                              | Share raw source code, early-stage, internal       | `setup.py sdist`                           |
| **2**  | **Wheel (Binary Package)**                      | `.whl`                                 | Standard modern Python library packaging           | `python -m build`, `setuptools`, `wheel`   |
| **3**  | **pip-installable Package**                     | Published `.whl` or `.tar.gz`          | Distribute via PyPI or private index               | `twine`, `build`, `pypi`                   |
| **4**  | **Editable Install**                            | Linked local path                      | Development mode (no rebuilds needed)              | `pip install -e .`                         |
| **5**  | **Docker Container Image**                      | Docker image layer                     | For production deployment / CI/CD                  | `docker build`, `buildx`                   |
| **6**  | **Standalone Executable**                       | `.exe`, `.app`, `.bin`                 | Offline or GUI applications                        | `pyinstaller`, `cx_Freeze`, `nuitka`       |
| **7**  | **Conda Package**                               | `.conda` / `.tar.bz2`                  | Data science or native binary dependency packaging | `conda build`, `conda-pack`                |
| **8**  | **Compiled Extension**                          | `.pyd`, `.so`, `.dll`                  | C/C++/CUDA performance modules                     | `cython`, `pybind11`, `maturin`            |
| **9**  | **ZIP Application**                             | `.pyz`                                 | Lightweight CLI or lambda-style app                | `python -m zipapp`                         |
| **10** | **Virtual Env Snapshot**                        | `requirements.txt` / `environment.yml` | Recreate experiment environments                   | `pip freeze`, `conda env export`           |
| **11** | **Serverless Deployment Package**               | `.zip` (handler + deps)                | AWS Lambda, Cloud Run, Azure Functions             | `AWS SAM`, `Zappa`, `Serverless Framework` |
| **12** | **Hybrid / Frozen Build (Executable + Docker)** | Docker + embedded binary               | Edge / secure or air-gapped ML deployment          | `pyinstaller` + `docker build`             |


# Grouped by Category

## Library Packaging (Reusability)
- Source Distribution  
- Wheel  
- pip-installable (PyPI/private)  
- Editable install  

Used when you want others to **import your code as a library**.

---

## Application Packaging (Execution)
- Docker Image  
- Executable (PyInstaller)  
- ZIP Application  
- Serverless ZIP  

Used when you want to **run your code as a service or application**.

---

## Environment & Native Builds (Reproducibility / Performance)
- Conda package  
- Virtual environment snapshot  
- Compiled extension (Cython/Rust)  
- Hybrid frozen builds  

Used for **machine learning, computer vision, or performance-sensitive reproducible pipelines**.

---

## Simplified Mental Model

| Goal | Build Type |
|------|-------------|
| Share code with others | Wheel / Source Distribution |
| Develop locally | Editable install |
| Deploy service or API | Docker image |
| Ship standalone app | PyInstaller / ZipApp |
| Run in cloud | Docker or Lambda ZIP |
| Recreate training environment | `requirements.txt` / Conda environment |
| Optimize performance | Cython / Rust extension |
