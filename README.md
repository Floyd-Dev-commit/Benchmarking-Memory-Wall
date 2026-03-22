# Benchmarking Memory Wall: Pandas vs. Dask vs. Polars

A rigorous, telemetry-driven performance evaluation of eager vs. lazy evaluation frameworks under strict single-node memory constraints. 

## 1. Dataset Preparation
This project utilizes the **New York City Taxi and Limousine Commission (NYC TLC) Trip Record Data**. 

### Source
* **Kaggle Mirror (Recommended):** You can download a pre-packaged mirror of the Parquet files from [Kaggle: TLC Trip Record Data - Yellow Taxi](https://www.kaggle.com/datasets/marcbrandner/tlc-trip-record-data-yellow-taxi).
* **Official Source:** Alternatively, download the raw monthly `.parquet` files directly from the [NYC TLC Official Website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

### Data Synthesis & Merging
To test out-of-core scalability, the dataset must be synthesized from monthly files into gigabyte-scale yearly files. 
* **Monthly to Yearly:** Monthly files are concatenated to form single-year datasets (e.g., `2019.parquet`).
* **Multi-Year Stress Test:** The maximum stress test dataset (`2009_2010.parquet`, 11.36 GB) is created by merging two yearly datasets.

**Schema Drift Handling:** Over the years, the TLC dataset changed column names (e.g., `Passenger Count` to `passenger_count`) and data types. We use Polars' `diagonal_relaxed` concatenation in the `merging_dataset.py` script to resolve this automatically.

---

## 2. Core Methodology & Scripts

Standard benchmarking scripts often fail to account for OS-level interference and architectural differences. This repository implements several advanced data engineering techniques to ensure empirical accuracy.

### A. OS Page Cache Eviction (Strict Cold-Start)
To prevent the OS from caching the Parquet files in RAM (which artificially inflates subsequent read speeds), we apply memory pressure before each test.
```python
# snippet from telemetry.py
def clear_os_cache():
    """Forces the OS to evict file caches by allocating 80% of available RAM."""
    mem = psutil.virtual_memory()
    target_bytes = int(mem.available * 0.8)
    try:
        _dummy_data = b'0' * target_bytes
        del _dummy_data
    except MemoryError:
        pass
    gc.collect()
    time.sleep(2)
```

### B. Dynamic Metadata Discovery
To avoid `KeyError` exceptions caused by historical schema drift, the pipeline lazily scans the Parquet schema before execution to extract exact column names without loading the payload.
```python
# snippet from telemetry.py
schema = pl.scan_parquet(file_path).collect_schema()
# Case-insensitive search
exact_col = next(col for col in schema.names() if col.lower() == target_col.lower())
```

### C. Telemetry Routing for Memory-Mapped I/O (mmap)
Standard process trackers (`psutil.Process().io_counters()`) cannot detect Polars' disk reads because it uses Memory-Mapped files (bypassing syscalls). Our pipeline routes telemetry based on the framework:
* **Pandas/Dask:** Process-level tracking.
* **Polars:** System-wide disk counters with background noise calibration.

### D. OS-Level Process Isolation & Polling
To prevent memory fragmentation and "high-water mark" contamination between tests, every framework is executed in an isolated Python subprocess via the `multiprocessing` module. For Dask, a background thread recursively polls the Resident Set Size (RSS) of the entire child-process tree to capture accurate cluster-wide memory usage.

---

## 3. How to Reproduce

### Prerequisites
* Python 3.11+
* 16 GB RAM minimum (to observe the Pandas OOM threshold)

### Step 1: Install Dependencies

Please check the main.py and Merge_Dataset.ipynb files to see the requirements.

### Step 2: Prepare the Data

Please use Merge_Dataset.ipynb to merge the downloaded datasets, you can find information about how to use it from comments in the ipynb file. 
Due to the large size of datasets, I didn't upload them to GitHub.
*If you have any questions, please contact me via floydstudy@outlook.com*

### Step 3: Run the Benchmark Suite
Execute the main testing pipeline. **Note:** Do not run this inside a Jupyter Notebook, as interactive kernels conflict with the `spawn` multiprocessing context required for strict memory isolation. Run it as a standalone script:
```bash
python main.py
```
The script will output terminal logs and generate a `.html` report of the telemetry metrics.

---

## 4. Hardware Environment
The benchmarks recorded in the report were executed on the following local workstation:
* **Processor:** Intel Core i7-1160G7 (8 Cores / 16 Threads)
* **RAM:** 16 GB (LPDDR4, 4266 MHz)
* **Storage:** 1TB NVMe SSD (PCI-e 4.0 x4)
* **OS:** Windows 10

---

**Author:** BIN XU  

**Email:** floydstudy@outlook.com

**Time Stamp:** 22, March, 2026

*For detailed discussions, please refer to the FULL PDF Report.
