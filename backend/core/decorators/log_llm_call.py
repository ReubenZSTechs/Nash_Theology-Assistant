from functools import wraps
from time import perf_counter
from typing import Optional, Any, Dict


from backend.core.observability.schema import LLMCallSchema
from backend.core.observability.record_call import CONFIG as log_file_config
from backend.core.observability.record_call import get_utc_time_now, write_record


def get_model_name(agent_config: Any) -> str:
    model_config = getattr(agent_config, "model_config", None) or {}
    agent_id = model_config.get("agent", {}) if isinstance(model_config, dict) else {}

    return agent_id.get("base_model") or getattr(agent_config, "model_name", "") or "unknown"


def format_payload(agent_config: Any, node_name: str, status: str, streaming: bool) -> Dict[str, Any]:
    return LLMCallSchema(
        model_name=get_model_name(agent_config),
        node_name=node_name,
        timestamp=get_utc_time_now(),
        status=status,
        streaming=streaming
    )


def write_to_logs(payload: Dict[str, Any], error_msg: Optional[str]) -> None:
    if not error_msg:
        write_record(log_file_config['LLM_LOG'], payload)
    else:
        write_record(log_file_config['ERROR_LOG'], payload)


def log_process_call(func):

    @wraps(func)
    def wrapper(self, user_prompt, *args, **kwargs):
        start = perf_counter()
        status = "success"
        error_message = None

        try:
            return func(self, user_prompt, *args, **kwargs)
        
        except Exception as exec:
            status = "error"
            error_message = str(exec)
            raise

        finally:
            latency = perf_counter() - start
            payload = format_payload(self, func.__name__, status, False)
            payload['latency'] = round(latency, 5)

            if not error_message:
                usage = getattr(self, "last_usage", None)
                generated = getattr(usage, "completion_tokens", None)

                payload['prompt_tokens'] = getattr(usage, "prompt_tokens", None)
                payload['generation_tokens'] = generated
                payload['ttft'] = None
                payload['token_generation'] = round(generated / latency, 5)

            write_to_logs(payload, error_message)

    return wrapper


def log_stream_call(func):

    @wraps(func)
    def wrapper(self, user_prompt, *args, **kwargs):
        start = perf_counter()
        status = "success"
        error_message = None
        ttft = None

        try:
            for chunk in func(self, user_prompt, *args, **kwargs):
                if not ttft:
                    ttft = perf_counter() - start

                yield chunk

        except Exception as exec:
            status = "error"
            error_message = str(exec)
            raise

        finally:
            latency = perf_counter() - start
            payload = format_payload(self, "answer", status, streaming=True)
            payload['latency'] = latency

            if not error_message:
                usage = getattr(self, "last_usage", None)
                generated = getattr(usage, "completion_tokens", None)

                payload['prompt_tokens'] = getattr(usage, "prompt_tokens", None)
                payload['generation_tokens'] = generated
                payload['ttft'] = round(ttft, 5) if ttft else None

                generation_time = latency - ttft if ttft else latency
                payload['token_generation'] = round(generated / generation_time, 5)

            write_to_logs(payload, error_message)

    return wrapper