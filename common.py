"""
Common functions.
"""

from sqlalchemy import create_engine

def make_engine():
    "Return a sqlalchemy engine connection to explore database."
    return create_engine("postgresql+psycopg2://explore:Ln2bOYAVCG6utNUSaSZaIVMH@localhost/explore")

