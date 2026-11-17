import os
import sys

# We import upstream modules to test against them, and CPython would drop
# __pycache__ into third_party/ — ground rule 1 says leave that tree untouched.
sys.dont_write_bytecode = True

# Tests fork after touching a tokenizer; the warning is noise, not a finding.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
