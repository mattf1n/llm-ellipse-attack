import argparse, itertools as it, json
from glob import glob
from tqdm import tqdm
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=None)
args = parser.parse_args()

client = OpenAI()

logfile = "data/batch_log.jsonl"
with open(logfile) as file:
    logs = map(json.loads, file)
    finished_filenames = set(log["metadata"]["description"] for log in logs)

unfinished_filenames = set(glob("data/queries/*.jsonl")) - finished_filenames

for filename in tqdm(list(unfinished_filenames)[: args.n]):
    with open(filename, "rb") as file:
        input_file = client.files.create(file=file, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": filename},
    )
    with open(logfile, "a") as file:
        print(batch.model_dump_json(), file=file)
