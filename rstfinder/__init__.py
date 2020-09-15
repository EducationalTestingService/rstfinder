# Ensure there won't be logging complaints about no handlers being set
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())