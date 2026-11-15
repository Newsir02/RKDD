"""Recommended compact recipe; shares data, teachers, evaluation and checkpoints."""
import sys
from train import main


if __name__ == "__main__":
    main([*sys.argv[1:], "--method", "trd"])
