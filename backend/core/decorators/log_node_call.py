from functools import wraps
from time import perf_counter
from typing import Any, Callable, Dict, Optional

from backend.core.observability.schema import NodeCallSchema
from backend.core.observability.record_call import CONFIG as log_file_config
from backend.core.observability.record_call import get_utc_time_now, write_record

def format_payload(node_name: str, status: str) -> Dict[str, Any]:
    return NodeCallSchema(
        node_name=node_name,
        timestamp=get_utc_time_now(),
        status=status
    )


def write_to_logs(payload: Dict[str, Any], error_msg: Optional[str]) -> None:
    if not error_msg:
        write_record(log_file_config['NODE_LOG'], payload)
    else:
        write_record(log_file_config['ERROR_LOG'], payload)


def count_chars(text: str) -> Optional[int]:
    if not text:
        return None

    if isinstance(text, str):
        return len(text)

    if isinstance(text, dict):
        return sum(len(char) for char in text.values() if isinstance(char, str)) or None

    if isinstance(text, (list, tuple)):
        return sum(len(char) for char in text if isinstance(char, str)) or None

    return None


def log_node(node_name: str) -> Callable:

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs):
            state = next((arg for arg in args if isinstance(arg, dict)), None)

            start = perf_counter()
            status = "success"
            error_message = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result

            except Exception as exec:
                status = "error"
                error_message = str(exec)
                raise

            finally:
                latency = perf_counter() - start
                payload = format_payload(node_name, status)
                payload['latency'] = latency

                if not error_message:
                    payload['input_chars'] = count_chars(state)
                    payload['output_chars'] = count_chars(result)

                write_to_logs(payload, error_message)

        return wrapper

    return decorator