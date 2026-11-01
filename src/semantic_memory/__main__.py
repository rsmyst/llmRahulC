"""Entry point: python -m semantic_memory"""

import logging

for _name in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from .server import main

if __name__ == "__main__":
    main()
