import logging
from typing import List, Tuple, Dict, Optional

def set_up_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%H:%M:%S')