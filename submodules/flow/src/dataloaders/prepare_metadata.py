import ast

import pandas as pd
from tqdm import tqdm


# deprecated
def prepare_jsonl(
    jsonl_path,
    filename_col,
    caption_or_tags_col,
    bucket_col_list,
    ext_col=None,
    chunksize=1000,
):
    jsonl_pbar = tqdm(desc="Processing jsonl chunks to jsonl", unit_scale=chunksize)
    jsonl = []
    for df in tqdm(pd.read_json(jsonl_path, lines=True, chunksize=chunksize)):
        for idx, data in df.iterrows():
            metadata = {
                "filename": (
                    data[filename_col] + "." + data[ext_col]
                    if ext_col
                    else data[filename_col]
                ),
                "caption_or_tags": data[caption_or_tags_col],
                "buckets": [tuple(x) for x in data[[str(x) for x in bucket_col_list]]],
            }
            jsonl.append(metadata)

        jsonl_pbar.update(1)
    return jsonl
