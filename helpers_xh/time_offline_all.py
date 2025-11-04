import subprocess
import re
import itertools
import csv

# Base model path
# MODEL_PATH = "Qwen/Qwen3-8B"
MODEL_PATH = "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/"

# Parameters to vary
# INPUT_OUTPUT_LENS = [(1024, 1024), (4096, 4096), (8192, 8192)]
INPUT_OUTPUT_LENS = [(1024, 1024), (4096, 4096)]
# INPUT_OUTPUT_LENS = [(1024, 1024)]
# ATTN_BACKENDS = ["flashinfer", "fa3", "triton"]
ATTN_BACKENDS = ["fa3", "triton"]
# DETERMINISTIC_KERNEL = ["", "sglang", "10_25_triton"]
DETERMINISTIC_KERNEL = ["sglang", "10_25_triton"]

# Regex pattern to extract benchmark duration
DURATION_PATTERN = re.compile(r"Benchmark duration \(s\):\s+([\d.]+)")

# Command template
CMD_TEMPLATE_NORMAL = (
    "python3 -m sglang.bench_offline_throughput "
    "--model-path {model_path} "
    "--dataset-name random "
    "--random-range-ratio 1 "
    "--random-input-len {input_len} "
    "--random-output-len {output_len} "
    "--num-prompts 256 "
    "--skip-warmup "
    "--attention-backend {backend} "
    "--disable-radix-cache"
)
CMD_TEMPLATE_DETERMINISTIC = (
    "python3 -m sglang.bench_offline_throughput "
    "--model-path {model_path} "
    "--dataset-name random "
    "--random-range-ratio 1 "
    "--random-input-len {input_len} "
    "--random-output-len {output_len} "
    "--num-prompts 256 "
    "--skip-warmup "
    "--attention-backend {backend} "
    "--disable-radix-cache "
    "--enable-deterministic-inference "
    "--batch-invariant-mm-folder {kernel}"
)

results = []

# Generate all combinations
for input_output_len in INPUT_OUTPUT_LENS:
    for backend in ATTN_BACKENDS:
        for kernel in DETERMINISTIC_KERNEL:
            input_len, output_len = input_output_len
            print(f"\nRunning benchmark: input={input_len}, output={output_len}, backend={backend}, kernel={kernel}")
            if kernel:
                cmd = CMD_TEMPLATE_DETERMINISTIC.format(
                    model_path=MODEL_PATH,
                    input_len=input_len,
                    output_len=output_len,
                    backend=backend,
                    kernel=kernel
                )
            else:
                cmd = CMD_TEMPLATE_NORMAL.format(
                    model_path=MODEL_PATH,
                    input_len=input_len,
                    output_len=output_len,
                    backend=backend
                )
            try:
                completed = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                output = completed.stdout
                match = DURATION_PATTERN.search(output)
                if match:
                    duration = float(match.group(1))
                    results.append((backend, kernel, input_len, output_len, duration))
                    print(f"✅ Benchmark duration: {duration:.2f} s")
                else:
                    print("⚠️ Benchmark duration not found in output.")
                    results.append((backend, kernel, input_len, output_len, None))
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running benchmark: {e}")
                results.append((backend, kernel, input_len, output_len, None))

# Write results to TSV
output_file = "offline_timing/time_all_results.tsv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["backend", "kernel", "input len", "output len", "time"])
    for backend, kernel, input_len, output_len, duration in results:
        if kernel == "":
            kernel = "normal"
        writer.writerow([backend, kernel, input_len, output_len, duration if duration is not None else "Failed"])

print(f"\nResults saved to: {output_file}")
