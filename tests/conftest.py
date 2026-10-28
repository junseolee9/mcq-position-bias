import sys

# We import upstream modules to test against them, and CPython would drop
# __pycache__ into third_party/ — ground rule 1 says leave that tree untouched.
sys.dont_write_bytecode = True
