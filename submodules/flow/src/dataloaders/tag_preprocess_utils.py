import csv
from collections import defaultdict


def prune(tags: list, tree):
    # Filter nodes with no child nodes in the same set
    filtered_tags = [
        node for node in tags if not any(child in tags for child in tree[node])
    ]
    # random.shuffle(filtered_tags)
    # filtered_tags = sorted(filtered_tags)
    return filtered_tags


def create_tree(csv_path):
    tree = defaultdict(list)
    with open(csv_path, "rt") as csvfile:
        reader = csv.DictReader(csvfile)
        # next(reader) # id,antecedent_name,consequent_name,created_at,status
        for row in reader:
            if row["status"] == "active":
                tree[row["consequent_name"]].append(row["antecedent_name"])
    return tree
