import logging
import os

def setup_logging(
    log_file: str = "app.log",
    level: int = logging.INFO,
) -> None:
    # Configure logging once for the entire application.

    # Prevent duplicate handlers if called multiple times
    if logging.getLogger().handlers:
        return

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        filemode="a",
    )