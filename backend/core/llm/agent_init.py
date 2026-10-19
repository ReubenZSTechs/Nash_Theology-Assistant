from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from openai import OpenAI
from typing import Optional, Union

from backend.utils.read_agent_config import read_config
from backend.utils.read_agent_schema import read_schema
from backend.core.decorators.log_llm_call import log_process_call, log_stream_call
from backend.core.exceptions.AgentExceptions import SchemaValidationError, AgentCallError, LLMOutputTruncatedError

import yaml
import logging
import json
import threading

logger = logging.getLogger(__name__)


CONFIG = {
    "MODEL_BRIDGE": {
        'llm': {
            'URL': "https://localhost:15000/v1",
            'SERVED_MODEL_NAME': "llm"
        },

        'embedding': {
            'URL': "https://localhost:15001/v1",
            'SERVED_MODEl_NAME': "embedding"
        }
    },
    "DEFAULT_SETTINGS": {
        'TIMEOUT': 300,
        'MAX_RETRIES': 2,
        'MAX_TOKENS': 4096,
        'TEMPERATURE': 0.2,
        'TOP_P': 0.9,
        'MIN_P': 0.05,
        'FREQUENCY_PENALTY': 0.0,
        'PRESENCE_PENALTY': 0.0,
        'REPETITION_PENALTY': 1.2,
        'REPAIR_JSON_ATTEMPTS': 2,
        'REPAIR_TRUNCATION_ATTEMPTS': 2,
        'GUIDED_DECODING_BACKEND': "xgrammer"
    }
}

REQUIRED_AGENT_KEYS = ["model_type", "role"]


class LLMManager:
    def __init__(self):
        pass


    def read_config(self, agent_config: str) -> dict:
        return read_config(agent_config)


    def read_schema(self, schema_file: str) -> dict:
        return read_schema(schema_file)


class LLMAgent:
    def __init__(self, agent_config: str, agent_schema: str):
        self.manager = LLMManager()
        self._local_agent = threading.local()

        self.agent_config = self.manager.read_config(agent_config)
        self.agent_schema = self.manager.read_schema(agent_schema)

        if not self.agent_config:
            raise ValueError(f"Agent config returned empty content. Check for errors!")

        if not self.agent_schema:
            raise ValueError(f"Agent schema output returned empty content. Check for errors!")

        self.agent_id = self.agent_config.get("agent")
        missing_agent_field = [field for field in REQUIRED_AGENT_KEYS if field not in self.agent_config]
        if missing_agent_field:
            raise KeyError(f"Agent ID missing keys {" ".join(missing_agent_field)}")

        self.agent_system_prompt = self.agent_config.get("system_prompt")
        self.agent_generation_config = self.agent_config.get("generation")

        self.agent_role = self.agent_id['role']
        self.agent_type = self.agent_id['model_type']

        bridge = CONFIG['MODEL_BRIDGE'].get(self.agent_type)
        if not bridge:
            raise ValueError(f"No vLLM container bridge available for model type {self.agent_type} with role {self.agent_role}")

        self.agent_url = bridge['URL']
        self.agent_model = bridge['SERVED_MODEL_NAME']

        self.client = OpenAI(
            base_url=self.agent_url,
            api_key="not-needed",
            timeout=self.agent_generation_config.get("timeout", CONFIG['DEFAULT_SETTINGS']['TIMEOUT']),
            max_retries=self.agent_generation_config.get("max_retries", CONFIG['DEFAULT_SETTINGS']['MAX_RETRIES'])
        )

        self.last_usage = None
        logger.info(f"Initiated agent with role {self.agent_role}")


    @property
    def last_usage(self):
        return getattr(self._local_agent, "usage", None)

    @property
    def last_finish_reason(self):
        return getattr(self._local_agent, "finish_reason", None)

    @last_usage.setter
    def last_usage(self, value):
        self._local_agent.usage = value

    @last_finish_reason.setter
    def last_finish_reason(self, value):
        self._local_agent.finish_reason = value


    def format_msg_payload(self, user_prompt: str, error_throwback_msg: dict=None) -> list[dict[str, str]]:
        agent_input = [
            {
                'role': 'system',
                'content': self.agent_system_prompt
            }
        ]

        if error_throwback_msg:
            agent_input.extend(error_throwback_msg)

        agent_input.append(            
            {
                'role': 'user',
                'content': user_prompt
            }
        )

        return agent_input


    def format_generation_payload(self, extra_args: dict=None) -> dict[str, Union[str, float, int]]:
        generation_payload = {
            'max_tokens': self.agent_generation_config.get('max_tokens', CONFIG['DEFAULT_SETTINGS']['MAX_TOKENS']),
            'temperature': self.agent_generation_config.get('temperature', CONFIG['DEFAULT_SETTINGS']['TEMPERATURE']),
            'top_p': self.agent_generation_config.get('top_p', CONFIG['DEFAULT_SETTINGS']['TOP_P']),
            'frequency_penalty': self.agent_generation_config.get("frequency_penalty", CONFIG['DEFAULT_SETTINGS']['FREQUENCY_PENALTY']),
            'presence_penalty': self.agent_generation_config.get("presence_penalty", CONFIG['DEFAULT_SETTINGS']['PRESENCE_PENALTY']),
            'min_p': self.agent_generation_config.get("min_p", CONFIG['DEFAULT_SETTINGS']['MIN_P']),
            'repetition_penalty': self.agent_generation_config.get("repetition_penalty", CONFIG['DEFAULT_SETTINGS']['REPETITION_PENALTY']),
            'guided_json': self.agent_schema,
            'guided_decoding_backend': self.agent_generation_config.get("guided_decoding_backend", CONFIG['DEFAULT_SETTINGS']['GUIDED_DECODING_BACKEND'])
        }

        if extra_args:
            generation_payload.update(extra_args)

        return generation_payload


    def format_response_format(self) -> dict[str, Union[str, dict]]:
        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'name': self.agent_role,
                'schema': self.agent_schema,
                'strict': True
            }
        }
        return response_format


    def build_request_kwargs(self, msg_payload: dict, extra_args: dict) -> dict[str, Union[str, dict]]:
        request_input = {
            "model": self.agent_model,
            "messages": msg_payload,
        }

        response_input = self.format_response_format()
        request_input['response_format'] = response_input

        generation_input = self.format_generation_payload(extra_args)

        request_input.update(generation_input)

        return request_input


    def parse_output(self, raw_output) -> dict:
        cleaned = raw_output.strip()

        if cleaned.startswith('```'):
            cleaned = cleaned.split('```')[1]
            if cleaned.startswith('json'):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        return json.loads(cleaned)


    @log_process_call
    def call_llm(self, user_prompt: str, extra_args: Optional[dict]=None, error_throwback_msg: dict=None) -> str:
        msg_payload = self.format_msg_payload(user_prompt, error_throwback_msg)
        request_kwargs = self.build_request_kwargs(msg_payload, extra_args)

        try:
            response = self.client.chat.completions.create(**request_kwargs)

        except Exception as exep:
            error_msg = str(exep)

            if "context" in error_msg and ("length" in error_msg or "token" in error_msg):
                raise SchemaValidationError(f"Agent with role {self.agent_role} exceeded context window tokens (max_tokens = {request_kwargs.get("max_tokens")})") from exep

            raise AgentCallError(f"Agent with role {self.agent_role} failed to call LLM (timeout_limit: {self.agent_generation_config.get('timeout')}s). Error got type {type(exep).__name__}. Error details below\n\n{exep}")

        
        self.last_usage = response.usage
        self.last_finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content

        if self.last_finish_reason == 'length':
            logger.warning(f"Role '{self.role}' hit max_tokens; output is truncated")

        return content


    def generate_llm_response(self, user_prompt: str, extra_args: Optional[dict]=None) -> dict:
        json_attempt = 0
        truncation_attempt = 0

        error_throwback_msg = None
        last_error = None
        token_budget = self.agent_generation_config.get("max_tokens", CONFIG['DEFAULT_SETTINGS']['MAX_TOKENS'])

        if not isinstance(token_budget, int):
            raise TypeError(f"Agent with role {self.agent_role} is not set to int type. Got {type(token_budget).__name__}")

        while json_attempt <= self.agent_generation_config.get("repair_json_attempts", CONFIG['DEFAULT_SETTINGS']['REPAIR_JSON_ATTEMPATS']):
            llm_output = self.call_llm(user_prompt, extra_args, error_throwback_msg)

            # Error: truncated output; insufficient max_tokens
            if self.last_finish_reason == "length":
                last_error = LLMOutputTruncatedError(f"Output is cut is {len(llm_output)} characters with max_tokens parameter set to {token_budget}")

                if truncation_attempt >= self.agent_generation_config.get("repair_truncation_attempts", CONFIG['DEFAULT_SETTINGS']['REPAIR_TRUNCATION_ATTEMPTS']):
                    break

                truncation_attempt += 1
                token_budget = int(token_budget * 1.6)
                error_throwback_msg = None
                logger.warning(f"Agent with role {self.agent_role} has truncated output. Attempting to repair with increased tokens.\nNew token budget={token_budget} ; Attempt {truncation_attempt}")

                continue

                
            # Error: JSON output failed to decode to json
            try:
                return self.parse_output(llm_output)

            except json.JSONDecodeError as exep:
                last_error = exep
                json_attempt += 1
                logger.warning(f"Agent with role {self.agent_role} returned unparseable JSON on attempt {json_attempt}. Detailed error:\n{exep}")

                error_throwback_msg = [
                    {
                        'role': 'assistant',
                        'content': llm_output
                    },
                    {
                        'role': 'user',
                        'content': f"Failed output parsing to json. Error details are {exep}. Return the parseable raw JSON object following the structure schema"
                    }
                ]

            except AgentCallError as exep:
                last_error = exep
                logger.warning(f"Agent with role {self.agent_role} failed to call LLM server. Detailed error:\n{exep}")


        if isinstance(last_error, LLMOutputTruncatedError):
            raise last_error

        if isinstance(last_error, AgentCallError):
            raise last_error

        raise SchemaValidationError(f"Agent with role {self.agent_role} failed to produce parseable JSON after {json_attempt + 1} attempts. Last error:\n{last_error}")


    @log_stream_call
    def generate_llm_answer(self, user_prompt: str, extra_args: Optional[dict]=None):
        msg_payload = self.format_msg_payload(user_prompt)
        request_kwargs = self.build_request_kwargs(msg_payload, extra_args)

        request_kwargs['stream'] = True
        request_kwargs['stream_options'] = {'include_usage': True}

        stream = self.client.chat.completions.create(**request_kwargs)

        for chunk in stream:
            if chunk.usage:
                self.last_usage = chunk.usage

            if chunk.choices:
                delta = chunk.choices[0].delta.content
            else:
                delta = None

            if delta:
                yield delta