## 2024-04-03 - Lazy Import Overhead in Dataset __getitem__
**Learning:** PyTorch dataset __getitem__ methods in this codebase run frequently. Having lazy imports (e.g. `from scipy.signal import resample`) inside dataset preprocessing functions introduces measurable overhead (~8% slowdown on the function itself) due to Python's import machinery overhead being called in a tight loop during data loading.
**Action:** Always move static/guaranteed dependencies to the global scope of preprocessing modules to avoid per-sample execution overhead.
