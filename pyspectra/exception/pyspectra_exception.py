class PySpectraException(BaseException):
    """Custom exception for a pyspectra steganography."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
