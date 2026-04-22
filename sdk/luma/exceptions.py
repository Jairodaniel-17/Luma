class LumaError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class LumaAuthError(LumaError):
    def __init__(self, message: str):
        super().__init__(401, message)


class LumaForbiddenError(LumaError):
    def __init__(self, message: str):
        super().__init__(403, message)


class LumaNotFoundError(LumaError):
    def __init__(self, message: str):
        super().__init__(404, message)


class LumaConflictError(LumaError):
    def __init__(self, message: str):
        super().__init__(409, message)
