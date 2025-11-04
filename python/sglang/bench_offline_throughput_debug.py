"""
Benchmark the throughput in the offline mode.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (the same as bench_serving.py).

# Usage
## Sharegpt dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10

## Random dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024
"""

import argparse
import asyncio
import dataclasses
import inspect
import json
import logging
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np

from sglang.bench_serving import (
    DatasetRow,
    get_dataset,
    get_tokenizer,
    sample_random_requests,
    set_ulimit,
)
from sglang.lang.backend.runtime_endpoint import Runtime
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs
import pdb


@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "sharegpt"
    dataset_path: str = ""
    num_prompts: int = 1000
    sharegpt_output_len: Optional[int] = None
    sharegpt_context_len: Optional[int] = None
    random_input_len: int = 1024
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    gsp_num_groups: int = 64
    gsp_prompts_per_group: int = 16
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256
    seed: int = 1
    disable_ignore_eos: bool = False
    extra_request_body: Optional[str] = None
    apply_chat_template: bool = False
    profile: bool = False
    skip_warmup: bool = False
    do_not_exit: bool = False
    prompt_suffix: str = ""
    tokenize_prompt: bool = False  # prompt in format of ids
    stats_folder_name: str = "test"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--backend", type=str, default=BenchArgs.backend)
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="sharegpt",
            choices=["sharegpt", "random", "generated-shared-prefix"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--dataset-path", type=str, default="", help="Path to the dataset."
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process. Default is 1000.",
        )
        parser.add_argument(
            "--sharegpt-output-len",
            type=int,
            default=BenchArgs.sharegpt_output_len,
            help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
        )
        parser.add_argument(
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Number of input tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-output-len",
            type=int,
            default=BenchArgs.random_output_len,
            help="Number of output tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Range of sampled ratio of input/output length, "
            "used only for random dataset.",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of groups with shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-prompts-per-group",
            type=int,
            default=BenchArgs.gsp_prompts_per_group,
            help="Number of prompts per group of shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="System prompt length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Question length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Target length in tokens for outputs in generated-shared-prefix dataset",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")
        parser.add_argument(
            "--disable-ignore-eos",
            action="store_true",
            help="Disable ignore EOS token",
        )
        parser.add_argument(
            "--extra-request-body",
            metavar='{"key1": "value1", "key2": "value2"}',
            type=str,
            default=BenchArgs.extra_request_body,
            help="Append given JSON object to the request payload. You can use this to specify"
            "additional generate params like sampling params.",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Apply chat template",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with "
            "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batches.",
        )
        parser.add_argument(
            "--do-not-exit",
            action="store_true",
            help="Do not exit the program. This is useful for nsys profile with --duration and --delay.",
        )
        parser.add_argument(
            "--prompt-suffix",
            type=str,
            default="",
            help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
        )
        ## @xinhao
        parser.add_argument(
            "--tokenize-prompt",
            type=bool,
            default=True,
            help="Tokenize prompt before sending to server (ensure exact prompt length as specified)",
        )
        parser.add_argument(
            "--stats-folder-name",
            type=str,
            default="test",
            help="Folder name to save last hidden states",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def throughput_test_once(
    backend_name: str,
    backend,
    reqs: List[DatasetRow],
    ignore_eos: bool,
    extra_request_body: Dict,
    profile: bool,
    stats_folder_name: str
):
    measurement_results = {
        "backend": backend_name,
        "successful_requests": len(reqs),
        "total_latency": -1,
        "total_input_tokens": sum(r.prompt_len for r in reqs),
        "total_output_tokens": -1,
        "request_throughput": -1,
        "input_throughput": -1,
        "output_throughput": -1,
        "total_throughput": -1,
    }

    prompt = [r.prompt for r in reqs]
    sampling_params = [
        {
            "temperature": 0,
            "max_new_tokens": r.output_len,
            "ignore_eos": ignore_eos,
            **extra_request_body,
        }
        for r in reqs
    ]

    if profile:
        assert (
            "SGLANG_TORCH_PROFILER_DIR" in os.environ
        ), "Please set SGLANG_TORCH_PROFILER_DIR."
        os.makedirs(os.environ["SGLANG_TORCH_PROFILER_DIR"], exist_ok=True)
        backend.start_profile()

    st = time.perf_counter()
    # gen_out = backend.generate(prompt=prompt, sampling_params=sampling_params)
    gen_out = backend.generate(input_ids=prompt, sampling_params=sampling_params, return_logprob=True, logprob_start_len=0, return_hidden_states=True)
    latency = time.perf_counter() - st

    prefill_logprob_all_req = []
    prefill_token_idx_all_req = []
    for i in range(len(gen_out)):
        prefill_logprob_cur_req = []
        prefill_token_idx_cur_req = []
        for j in range(1, len(gen_out[i]['meta_info']['input_token_logprobs'])):
            prefill_logprob_cur_req.append(gen_out[i]['meta_info']['input_token_logprobs'][j][0])
            prefill_token_idx_cur_req.append(gen_out[i]['meta_info']['input_token_logprobs'][j][1])
        prefill_logprob_all_req.append(prefill_logprob_cur_req)
        prefill_token_idx_all_req.append(prefill_token_idx_cur_req)
    prefill_prob_all_req = np.exp(np.asarray(prefill_logprob_all_req))
    prefill_token_idx_all_req = np.asarray(prefill_token_idx_all_req)

    decode_logprob_all_req = []
    decode_token_idx_all_req = []
    for i in range(len(gen_out)):
        decode_logprob_cur_req = []
        decode_token_idx_cur_req = []
        for j in range(1, len(gen_out[i]['meta_info']['output_token_logprobs'])):
            decode_logprob_cur_req.append(gen_out[i]['meta_info']['output_token_logprobs'][j][0])
            decode_token_idx_cur_req.append(gen_out[i]['meta_info']['output_token_logprobs'][j][1])
        decode_logprob_all_req.append(decode_logprob_cur_req)
        decode_token_idx_all_req.append(decode_token_idx_cur_req)
    decode_prob_all_req = np.exp(np.asarray(decode_logprob_all_req))
    decode_token_idx_all_req = np.asarray(decode_token_idx_all_req)

    prefill_hidden_states_all_req = []
    decode_hidden_states_all_req = []
    for i in range(len(gen_out)):
        prefill_hidden_states_cur_req = gen_out[i]['meta_info']['hidden_states'][0]  # [p,f]
        decode_hidden_states_cur_req = gen_out[i]['meta_info']['hidden_states'][1:]  # [d-1,f]
        prefill_hidden_states_all_req.append(prefill_hidden_states_cur_req)
        decode_hidden_states_all_req.append(decode_hidden_states_cur_req)
    prefill_hidden_states_all_req = np.asarray(prefill_hidden_states_all_req)  # [r,p,f]
    decode_hidden_states_all_req = np.asarray(decode_hidden_states_all_req)    # [r,d-1,f]

    folder_name = stats_folder_name
    # folder_name = "11_03/normal/r128p512d512"
    # folder_name = "11_03/sglang_determ/r128p512d512"
    # folder_name = "11_03/our_rms_cuda/r128p512d512"
    os.makedirs(f'./stats_logprob/{folder_name}', mode=0o777, exist_ok=True)
    np.save(f'./stats_logprob/{folder_name}/prefill_prob_all_req.npy', prefill_prob_all_req)
    np.save(f'./stats_logprob/{folder_name}/prefill_token_idx_all_req.npy', prefill_token_idx_all_req)
    np.save(f'./stats_logprob/{folder_name}/prefill_hidden_states_all_req.npy', prefill_hidden_states_all_req)
    np.save(f'./stats_logprob/{folder_name}/decode_prob_all_req.npy', decode_prob_all_req)
    np.save(f'./stats_logprob/{folder_name}/decode_token_idx_all_req.npy', decode_token_idx_all_req)
    np.save(f'./stats_logprob/{folder_name}/decode_hidden_states_all_req.npy', decode_hidden_states_all_req)


    if profile:
        dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
        known_files = set(os.listdir(dir))
        backend.stop_profile()
        monitor_trace_file(known_files, dir)

    if backend_name == "runtime":
        gen_out = json.loads(gen_out)

    server_info = backend.get_server_info()

    measurement_results["total_latency"] = latency
    measurement_results["total_output_tokens"] = sum(
        o["meta_info"]["completion_tokens"] for o in gen_out
    )
    measurement_results["request_throughput"] = (
        measurement_results["successful_requests"] / latency
    )
    measurement_results["input_throughput"] = (
        measurement_results["total_input_tokens"] / latency
    )
    measurement_results["output_throughput"] = (
        measurement_results["total_output_tokens"] / latency
    )
    measurement_results["total_throughput"] = (
        measurement_results["total_input_tokens"]
        + measurement_results["total_output_tokens"]
    ) / latency

    if inspect.isawaitable(server_info):
        server_info = asyncio.run(server_info)

    measurement_results["last_gen_throughput"] = server_info["internal_states"][0][
        "last_gen_throughput"
    ]

    return measurement_results


def monitor_trace_file(known_files, directory, interval=1):
    print(f"Monitoring {directory} for new trace files...")

    while True:
        flag = False
        time.sleep(interval)
        current_files = set(os.listdir(directory))

        new_files = current_files - known_files
        for new_file in new_files:
            new_file_path = os.path.join(directory, new_file)
            print(f"New file detected: {new_file}")

            previous_size = 0
            while True:
                try:
                    current_size = os.path.getsize(new_file_path)
                except FileNotFoundError:
                    print(f"File {new_file} is no longer accessible.")
                    break

                if current_size > previous_size:
                    previous_size = current_size
                else:
                    flag = True
                    break

                time.sleep(interval)
        if flag:
            break


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    if bench_args.backend == "engine":
        backend = Engine(**dataclasses.asdict(server_args))
        if not backend:
            raise ValueError("Please provide valid engine arguments")
    elif bench_args.backend == "runtime":
        backend = Runtime(**dataclasses.asdict(server_args))
    else:
        raise ValueError('Please set backend to either "engine" or "runtime"')

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # Set global environments
    set_ulimit()
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # Parse args
    extra_request_body = {}
    if bench_args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    # Read dataset
    input_requests = get_dataset(bench_args, tokenizer)

    warmup_requests = sample_random_requests(
        input_len=256,
        output_len=16,
        num_prompts=min(bench_args.num_prompts, 16),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=bench_args.dataset_path,
    )

    # Warm up
    if not bench_args.skip_warmup:
        logging.info("\nWarmup...")
        throughput_test_once(
            backend_name=bench_args.backend,
            backend=backend,
            reqs=warmup_requests,
            ignore_eos=not bench_args.disable_ignore_eos,
            extra_request_body=extra_request_body,
            profile=False,
        )
        time.sleep(0.5)

    logging.info("\nBenchmark...")
    result = throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=input_requests,
        ignore_eos=not bench_args.disable_ignore_eos,
        extra_request_body=extra_request_body,
        profile=bench_args.profile,
        stats_folder_name=bench_args.stats_folder_name
    )
    backend.shutdown()

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    print(
        "\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=50, c="=")
    )
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print(
        "{:<40} {:<10}".format("Total generated tokens:", result["total_output_tokens"])
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Last generation throughput (tok/s):", result["last_gen_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", result["request_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", result["input_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", result["output_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", result["total_throughput"]
        )
    )
    print("=" * 50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    # handling ModelScope model downloads
    if os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() in ("true", "1"):
        if os.path.exists(args.model_path):
            print(f"Using local model path: {args.model_path}")
        else:
            try:
                from modelscope import snapshot_download

                print(f"Using ModelScope to download model: {args.model_path}")

                # download the model and replace args.model_path
                args.model_path = snapshot_download(
                    args.model_path,
                )
                print(f"Model downloaded to: {args.model_path}")
            except Exception as e:
                print(f"ModelScope download failed: {str(e)}")
                raise e

    server_args = ServerArgs.from_cli_args(args)
    server_args.enable_return_hidden_states = True  # @xinhao: for some reason must manully set it here. Cannot pass from cmd.
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    throughput_test(server_args, bench_args)

    while bench_args.do_not_exit:
        pass
