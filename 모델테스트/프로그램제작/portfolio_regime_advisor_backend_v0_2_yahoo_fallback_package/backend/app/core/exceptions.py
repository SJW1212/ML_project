class PRAError(Exception):
    """Base application exception."""


class ValidationError(PRAError):
    pass


class DataNotFoundError(PRAError):
    pass


class ProviderError(PRAError):
    pass


class ModelError(PRAError):
    pass
