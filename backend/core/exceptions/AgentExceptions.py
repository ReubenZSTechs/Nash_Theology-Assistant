class SchemaValidationError(ValueError):
    pass

class AgentNotFoundError(KeyError):
    pass

class DuplicateAgentRoleError(ValueError):
    pass

class AgentCallError(RuntimeError):
    pass

class LLMOutputTruncatedError(RuntimeError):
    def __init__(self, message: str, partial_output: str):
        super().__init__(message)
        self.partial_output = partial_output