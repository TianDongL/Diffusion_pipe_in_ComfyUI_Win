import sys
import os

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataloaders.tag_preprocess_utils import create_tree, prune


tree = create_tree("tag_implications-2024-06-16.csv")


pruned = prune(["canine", "fox", "zebra", "equine"], tree)
print(pruned)
