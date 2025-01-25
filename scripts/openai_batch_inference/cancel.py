import argparse, itertools as it, json
from openai import OpenAI

client = OpenAI()

logfile = "data/batch_log.jsonl"
with open(logfile) as file:
    logs = list(map(json.loads, file))

for log in logs:
    if "batch" in log["metadata"]["description"]:
        client.batches.cancel(log["id"])
