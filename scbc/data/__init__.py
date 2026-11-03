"""Data loading for CMU-MOSEI (segment level) and IEMOCAP-4 (dialogue level)."""
from .iemocap import load_dialogues
from .mosei import load_split

__all__ = ["load_split", "load_dialogues"]
