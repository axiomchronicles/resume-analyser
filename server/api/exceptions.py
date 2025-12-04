

class ResumeAnalserBaseException(Exception):
    def __init__(self, details: str | None, status: int = 500, headers: dict | None = None):
        self.details = details
        self.status = status
        self.headers = headers


class ApiResponseError(ResumeAnalserBaseException):
    pass