"""Full SCBC models: a benchmark backbone with the debt book / corrector / objective
chain built on top of it."""
from .iemocap import IemocapSCBC
from .mosei import MoseiSCBC

__all__ = ["MoseiSCBC", "IemocapSCBC"]
