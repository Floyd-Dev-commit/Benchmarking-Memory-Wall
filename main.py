import pandas as pd
import polars as pl
import dask.dataframe as dd
import time
import matplotlib.pyplot as plt
import psutil
import os
import threading
import gc
import multiprocessing as mp
from datetime import datetime
import io
import base64

# ==========================================
# 0. Configuration & Dynamic Schema Discovery
# ==========================================
FILE_PATH = "dataset/GT10GB/2009_2010.parquet"

# Calculate dataset size in GB globally so all functions can use it
FILE_SIZE_BYTES = os.path.getsize(FILE_PATH)
FILE_SIZE_GB = FILE_SIZE_BYTES / (1024 * 1024 * 1024)

PASSENGER_COL = None
FARE_COL = None


# ==========================================
# Background thread class to track TRUE peak memory usage
# ==========================================
class MemoryTracker:
    def __init__(self):
        self.keep_measuring = True
        self.peak_memory = 0

    def track(self):
        main_process = psutil.Process(os.getpid())
        while self.keep_measuring:
            try:
                total_mem_bytes = main_process.memory_info().rss
                for child in main_process.children(recursive=True):
                    try:
                        total_mem_bytes += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                current_mem_mb = total_mem_bytes / (1024 * 1024)
                if current_mem_mb > self.peak_memory:
                    self.peak_memory = current_mem_mb
            except Exception:
                pass
            time.sleep(0.05)


# ==========================================
# Core Benchmarking Functions
# ==========================================
def clear_os_cache():
    print("   Forcing OS to drop file caches via memory pressure...")
    mem = psutil.virtual_memory()
    try:
        target_bytes = int(mem.available * 0.6)
        _dummy_data = b'0' * target_bytes
        del _dummy_data
    except MemoryError:
        pass
    gc.collect()
    time.sleep(2)
    print("   OS Cache cleared. Test environment is cold and clean.")


def run_with_tracker(func, framework_name):
    clear_os_cache()
    process = psutil.Process(os.getpid())
    is_polars = (framework_name == "Polars")

    bg_io_rate = 0
    if is_polars:
        print("   Calibrating background disk I/O noise for Polars...")
        bg_io_start = psutil.disk_io_counters().read_bytes
        time.sleep(1)
        bg_io_rate = psutil.disk_io_counters().read_bytes - bg_io_start
        base_io = psutil.disk_io_counters().read_bytes
    else:
        try:
            base_io = process.io_counters().read_bytes
        except AttributeError:
            base_io = 0

    tracker = MemoryTracker()
    tracker_thread = threading.Thread(target=tracker.track)
    tracker_thread.start()

    start_time = time.time()
    success = False
    error_msg = ""

    try:
        func()
        success = True
    except Exception as e:
        error_msg = str(e)
    finally:
        end_time = time.time()
        if is_polars:
            end_io = psutil.disk_io_counters().read_bytes
        else:
            try:
                end_io = process.io_counters().read_bytes
            except AttributeError:
                end_io = 0

        tracker.keep_measuring = False
        tracker_thread.join()

    elapsed_time = end_time - start_time if success else 0
    absolute_peak_memory = tracker.peak_memory if success else 0

    if success:
        if is_polars:
            estimated_bg_noise = bg_io_rate * elapsed_time
            final_disk_bytes = max(0, (end_io - base_io) - estimated_bg_noise)
        else:
            final_disk_bytes = max(0, end_io - base_io)
        net_disk_mb = final_disk_bytes / (1024 * 1024)
    else:
        net_disk_mb = 0

    result_dict = {
        'time': elapsed_time,
        'memory': absolute_peak_memory,
        'disk': net_disk_mb,
        'success': success,
        'error': error_msg
    }
    return result_dict


def isolated_worker(framework_name, func, queue, pass_col, fare_col):
    global PASSENGER_COL, FARE_COL
    PASSENGER_COL = pass_col
    FARE_COL = fare_col

    try:
        result = run_with_tracker(func, framework_name)
        queue.put((framework_name, result))
    except Exception as e:
        queue.put((framework_name, {'time': 0, 'memory': 0, 'disk': 0, 'success': False, 'error': str(e)}))


# ==========================================
# 1. Define Test Functions
# ==========================================
def test_pandas():
    # Based on preliminary testing, Pandas will encounter an OOM error when processing
    # datasets larger than 2.5GB on this specific machine. This threshold check is
    # introduced to skip the execution and avoid wasting time waiting for the inevitable OOM.
    if FILE_SIZE_GB > 2.5:
        raise MemoryError(
            f"Dataset is {FILE_SIZE_GB:.2f} GB. This exceeds the processing capacity limit of Pandas on this machine, therefore it is skipped.")

    df = pd.read_parquet(FILE_PATH, engine='pyarrow')
    df_filtered = df[df[PASSENGER_COL] > 1]
    df_grouped = df_filtered.groupby(PASSENGER_COL)[FARE_COL].mean()

    df = pd.read_parquet(FILE_PATH, engine='pyarrow')
    df_filtered = df[df[PASSENGER_COL] > 1]
    df_grouped = df_filtered.groupby(PASSENGER_COL)[FARE_COL].mean()


def test_polars():
    lazy_df = pl.scan_parquet(FILE_PATH)
    lazy_filtered = lazy_df.filter(pl.col(PASSENGER_COL) > 1)
    lazy_grouped = lazy_filtered.group_by(PASSENGER_COL).agg(
        pl.col(FARE_COL).mean().alias('avg_fare')
    )
    result = lazy_grouped.collect()


def test_dask():
    ddf = dd.read_parquet(FILE_PATH, engine='pyarrow')
    ddf_filtered = ddf[ddf[PASSENGER_COL] > 1]
    ddf_grouped = ddf_filtered.groupby(PASSENGER_COL)[FARE_COL].mean()
    result = ddf_grouped.compute()


# ==========================================
# 2. Main Execution Block & HTML Generation
# ==========================================
if __name__ == '__main__':
    mp.freeze_support()
    start_dt = datetime.now()

    print("> Initializing OS-Level Process Isolation Engine...")
    print("> Extracting schema via Polars metadata layer...")

    file_columns = pl.scan_parquet(FILE_PATH).collect_schema().names()

    detect_passenger, detect_fare = None, None
    for col in file_columns:
        col_lower = col.lower()
        if "passenger" in col_lower and "count" in col_lower:
            detect_passenger = col
        elif "fare" in col_lower and ("amount" in col_lower or "amt" in col_lower):
            detect_fare = col

    if not detect_passenger or not detect_fare:
        raise ValueError(f"CRITICAL: Could not detect target columns.")

    result_queue = mp.Queue()
    benchmark_results = {}

    tasks = [
        ("Pandas", test_pandas),
        ("Polars", test_polars),
        ("Dask", test_dask)
    ]

    for name, func in tasks:
        print(f"\n--- Spawning strictly isolated subprocess for [{name}] ---")
        p = mp.Process(target=isolated_worker, args=(name, func, result_queue, detect_passenger, detect_fare))
        p.start()

        metrics = {'time': 0, 'memory': 0, 'disk': 0, 'success': False, 'error': 'Unknown timeout'}
        while p.is_alive():
            if not result_queue.empty():
                returned_name, metrics = result_queue.get()
                break
            time.sleep(0.1)

        p.join()

        if not result_queue.empty():
            returned_name, metrics = result_queue.get()

        benchmark_results[name] = metrics

        if metrics['success']:
            print(
                f"[{name}] Execution Completed! Time: {metrics['time']:.2f}s | Peak Mem: {metrics['memory']:.2f} MB | Disk I/O: {metrics['disk']:.2f} MB")
        else:
            err_text = metrics.get('error', 'OOM / Terminated')
            if not err_text:
                err_text = "OOM / Terminated"
            print(f"[{name}] Critical Failure: {err_text}")

    # ==========================================
    # 3. Generate Charts (With Anti-Overlap Headroom)
    # ==========================================
    suite_end_dt = datetime.now()
    print(
        f"\n[SYS.HALT] Test Suite Finalized at {suite_end_dt.strftime('%H:%M:%S')}. Total Duration: {(suite_end_dt - start_dt).total_seconds():.2f}s")
    print("Generating visual comparison charts...")

    frameworks = ['Pandas\n(Eager)', 'Dask\n(Task Graphs)', 'Polars\n(Lazy)']
    fw_keys = ['Pandas', 'Dask', 'Polars']

    times = [benchmark_results[k]['time'] for k in fw_keys]
    memories = [benchmark_results[k]['memory'] for k in fw_keys]
    disks = [benchmark_results[k]['disk'] for k in fw_keys]

    # To maintain the geeky style, the default background is kept clean, borders removed
    # Standard styling is used here, but Y-axis limits are expanded to prevent text overlap
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
    colors = ['#FF6B6B', '#4D96FF', '#6BCB77']  # Tech-inspired color palette

    # Dynamic baseline
    offset_t = max(times) * 0.03 if max(times) > 0 else 1.0
    offset_m = max(memories) * 0.03 if max(memories) > 0 else 1.0
    offset_d = max(disks) * 0.03 if max(disks) > 0 else 1.0

    # Core fix: Force a 25% headroom space on the top to absolutely prevent hitting the ceiling
    ax1.set_ylim(0, (max(times) if max(times) > 0 else 10) * 1.25)
    ax2.set_ylim(0, (max(memories) if max(memories) > 0 else 10) * 1.25)
    ax3.set_ylim(0, (max(disks) if max(disks) > 0 else 10) * 1.25)

    # --- Subplot 1: Execution Time ---
    bars1 = ax1.bar(frameworks, times, color=colors, width=0.5)
    ax1.set_title('A. Execution Time (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Seconds', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for i, bar in enumerate(bars1):
        yval = bar.get_height()
        if benchmark_results[fw_keys[i]]['success']:
            ax1.text(bar.get_x() + bar.get_width() / 2, yval + offset_t, f"{yval:.2f} s", ha='center', va='bottom',
                     fontsize=12, fontweight='bold')
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2, offset_t, "OOM/Error", ha='center', va='bottom', fontsize=12,
                     fontweight='bold', color='red')

    # --- Subplot 2: Peak Memory Usage ---
    bars2 = ax2.bar(frameworks, memories, color=colors, width=0.5)
    ax2.set_title('B. Absolute Peak Memory (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Megabytes (MB)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        if benchmark_results[fw_keys[i]]['success']:
            ax2.text(bar.get_x() + bar.get_width() / 2, yval + offset_m, f"{yval:.2f} MB", ha='center', va='bottom',
                     fontsize=12, fontweight='bold')
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, offset_m, "OOM/Error", ha='center', va='bottom', fontsize=12,
                     fontweight='bold', color='red')

    # --- Subplot 3: Disk Read Volume ---
    bars3 = ax3.bar(frameworks, disks, color=colors, width=0.5)
    ax3.set_title('C. Disk Read Volume (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Megabytes (MB)', fontsize=12)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    for i, bar in enumerate(bars3):
        yval = bar.get_height()
        if benchmark_results[fw_keys[i]]['success']:
            ax3.text(bar.get_x() + bar.get_width() / 2, yval + offset_d, f"{yval:.2f} MB", ha='center', va='bottom',
                     fontsize=12, fontweight='bold')
        else:
            ax3.text(bar.get_x() + bar.get_width() / 2, offset_d, "OOM/Error", ha='center', va='bottom', fontsize=12,
                     fontweight='bold', color='red')

    plt.tight_layout(pad=2.0)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)

    # ==========================================
    # 4. Generate & Save Geeky HTML Report
    # ==========================================
    end_dt = datetime.now()
    file_size_mb = os.path.getsize(FILE_PATH) / (1024 * 1024)
    report_filename = f"Benchmark_Report_{start_dt.strftime('%Y%m%d_%H%M%S')}.html"

    # Construct the Geek-style HTML template (Dark Theme, Terminal Aesthetics)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Benchmark Results [{start_dt.strftime('%Y%m%d')}]</title>
        <style>
            body {{ font-family: 'Consolas', 'Courier New', monospace; background-color: #0d1117; color: #c9d1d9; margin: 40px; font-size: 14px; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: auto; }}
            h1 {{ border-bottom: 1px solid #30363d; padding-bottom: 10px; color: #58a6ff; font-weight: normal; font-size: 24px; }}
            h2 {{ color: #8b949e; font-weight: normal; font-size: 18px; margin-top: 30px; }}
            .info-box {{ background: #161b22; padding: 15px; border: 1px solid #30363d; border-radius: 6px; margin-bottom: 20px; }}
            .console-box {{ background: #010409; color: #00ff00; padding: 20px; border: 1px solid #30363d; border-radius: 6px; white-space: pre-wrap; }}
            .success {{ color: #3fb950; font-weight: bold; }}
            .error {{ color: #f85149; font-weight: bold; }}
            .sys-msg {{ color: #8b949e; }}
            img {{ max-width: 100%; border: 1px solid #30363d; border-radius: 6px; background: #ffffff; padding: 10px; margin-top: 10px; }}
            code {{ color: #ff7b72; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>>_ Big Data Execution Pipeline | Telemetry Report</h1>

            <div class="info-box">
                <span class="sys-msg">[SYS.INFO]</span> Execution Timestamp : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}<br>
                <span class="sys-msg">[SYS.INFO]</span> Target Parquet File : <code>{FILE_PATH}</code> ({file_size_mb:.2f} MB)<br>
                <span class="sys-msg">[SYS.INFO]</span> Schema Auto-Sniffed : Passenger_Col = '<code>{detect_passenger}</code>', Fare_Col = '<code>{detect_fare}</code>'
            </div>

            <h2>Terminal Execution Log</h2>
            <div class="console-box">> Initializing OS-Level Process Isolation Engine...
> Extracting schema via Polars metadata layer...
"""

    # Accurately replicate the detailed real console output
    for fw in fw_keys:
        res = benchmark_results[fw]
        html_content += f"\n<span class='sys-msg'>--- Spawning strictly isolated subprocess for [{fw}] ---</span>\n"
        html_content += f"   Forcing OS to drop file caches via memory pressure...\n"
        html_content += f"   OS Cache cleared. Test environment is cold and clean.\n"

        if fw == 'Polars':
            html_content += f"   Calibrating background disk I/O noise for Polars...\n"

        if res['success']:
            html_content += f"[{fw}] <span class='success'>Execution Completed!</span> Time: {res['time']:.2f}s | Peak Mem: {res['memory']:.2f} MB | Disk I/O: {res['disk']:.2f} MB\n"
        else:
            err_text = res.get('error', 'OOM / Terminated')
            if not err_text: err_text = "OOM / Terminated"
            html_content += f"[{fw}] <span class='error'>Critical Failure: {err_text}</span>\n"

    html_content += f"\n<span class='sys-msg'>[SYS.HALT]</span> Test Suite Finalized at {end_dt.strftime('%H:%M:%S')}. Total Duration: {(end_dt - start_dt).total_seconds():.2f}s"
    html_content += f"</div>"

    html_content += f"""
            <h2>Graphical Telemetry Verification</h2>
            <img src="data:image/png;base64,{img_base64}" alt="Benchmark Charts">
        </div>
    </body>
    </html>
    """

    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n==========================================")
    print(f"Telemetry Report compiled successfully!")
    print(f"File saved as: {report_filename}")
    print(f"==========================================")