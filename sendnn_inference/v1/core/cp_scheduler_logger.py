"""Spyre chunked prefill scheduler logging. Collects state at each step"""
import json
import os
import time
from datetime import datetime
from typing import Any

from vllm.v1.outputs import ModelRunnerOutput

import sendnn_inference.envs as envs


def create_cp_scheduler_logger(max_model_len: int, max_num_seqs: int, block_size: int):
    if envs.SENDNN_INFERENCE_CP_SCHEDULER_LOGGING_ENABLED == 1:
        return CPSchedulerLogger(max_model_len, max_num_seqs, block_size)
    return CPSchedulerLoggerBase()


class CPSchedulerLoggerBase:
    """A no-op base class for use when logging is disabled"""

    def __init__(self):
        pass

    def __del__(self):
        pass

    def log(
        self,
        model_runner_output: ModelRunnerOutput,
        waiting: list[Any],
        running: list[Any],
        tkv: int,
    ):
        pass


class CPSchedulerLogger(CPSchedulerLoggerBase):
    """A chunked prefill logging object"""

    def __init__(self, max_model_len: int, max_num_seqs: int, block_size: int):
        super().__init__()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(
            envs.SENDNN_INFERENCE_CP_SCHEDULER_LOGGING_DIR,
            f"cp_scheduler_logging_{timestamp}.jsonl",
        )

        first_line = {
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "block_size": block_size,
        }
        json_data_line = json.dumps(first_line)
        with open(self.log_path, "a") as f:
            f.write(json_data_line + "\n")

    def log(
        self,
        model_runner_output: ModelRunnerOutput,
        waiting: list[Any],
        running: list[Any],
        tkv: int,
    ):
        data: dict[str, Any] = {}
        # metadata
        data["logging_time"] = time.time()
        data["tkv"] = tkv
        # waiting list
        data["waiting"] = {}
        data["waiting"]["id"] = [w.request_id for w in waiting]
        data["waiting"]["arrival_time"] = [w.arrival_time for w in waiting]
        # running list
        data["running"] = {}
        data["running"]["id"] = [r.request_id for r in running]
        data["running"]["arrival_time"] = [r.arrival_time for r in running]
        data["running"]["prompt_len"] = [r.num_prompt_tokens for r in running]
        data["running"]["max_tokens"] = [r.max_tokens for r in running]
        data["running"]["computed_tokens"] = [r.num_computed_tokens for r in running]
        data["running"]["output_tokens"] = [r.num_output_tokens for r in running]

        # also consider output tokens that were just computed
        req_to_index = model_runner_output.req_id_to_index
        for req in model_runner_output.req_ids:
            step_num_output_tokens = len(model_runner_output.sampled_token_ids[req_to_index[req]])
            index = data["running"]["id"].index(req)
            data["running"]["output_tokens"][index] += step_num_output_tokens

        json_data_line = json.dumps(data)
        with open(self.log_path, "a") as f:
            f.write(json_data_line + "\n")

# Made with Bob
