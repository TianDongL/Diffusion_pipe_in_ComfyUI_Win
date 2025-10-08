import sys
import os
import json

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataloaders.bucketing_logic import create_bucket_column_pandas
from src.dataloaders.prepare_metadata import prepare_jsonl
from src.dataloaders.tag_preprocess_utils import prune, create_tree
from src.dataloaders.utils import save_as_jsonl, read_jsonl
from tqdm import tqdm
import random

INPUT_CSV = "furry_50k_4o/filtered-posts-2024-06-16_with_4o_captions.csv"
OUTPUT_JSONL = "post_truncated_bucket.jsonl"
BUCKET_RES = [384, 512, 640, 768, 896, 1024]

# remove the cache
if os.path.exists(OUTPUT_JSONL):
    os.remove(OUTPUT_JSONL)

# prepare standarized jsonl for training
create_bucket_column_pandas(
    INPUT_CSV,
    OUTPUT_JSONL,
    BUCKET_RES,
    step=8,
    ratio_cutoff=2,
    height_col_name="image_height",
    width_col_name="image_width",
    chunksize=1000,
)
jsonl = prepare_jsonl(
    OUTPUT_JSONL,
    filename_col="md5",
    caption_or_tags_col="tag_string",
    bucket_col_list=BUCKET_RES,
    ext_col="file_ext",
)

save_as_jsonl(jsonl, "test.jsonl")


# preprocess in dataloader

jsonl = read_jsonl("test.jsonl")

implication_tree = create_tree("tag_implications-2024-06-16.csv")


buckets = {}
for i in tqdm(range(len(jsonl))):
    caption_or_tags = prune(jsonl[i]["caption_or_tags"].split(" "), implication_tree)
    res_bucket = tuple(random.choice(jsonl[i]["buckets"]))  # use seed!!

    # this is already in standarized format
    sample = {
        "filename": jsonl[i]["filename"],
        "caption_or_tags": jsonl[i]["caption_or_tags"],
        "bucket": res_bucket,
    }

    if res_bucket in buckets:
        buckets[res_bucket].append(sample)
    else:
        buckets[res_bucket] = [sample]


batch_size = 20
# pre-shuffle bucket for maximum randomness :P
for key in tqdm(buckets.keys(), desc="shuffling buckets"):
    random.shuffle(buckets[key])

# simple post counter
post_count = 0
for k in buckets:
    post_count += len(buckets[k])
print(f"There are {post_count} text image pairs.")

# arrange into batches
batches = []
for b in buckets:
    samples = []
    # if not full batch drop the last batch to prevent bad things
    for s in buckets[b]:
        if len(samples) < batch_size:
            samples.append(s)
        elif len(samples) == batch_size:
            batches.append(samples)
            samples = [s]
print(f"We got {len(batches)} batches from these pairs.")

random.shuffle(batches)


print()
