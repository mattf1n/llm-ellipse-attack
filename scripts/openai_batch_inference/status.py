import argparse, itertools as it, json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

logfile = "data/batch_log.jsonl"
with open(logfile) as file:
    logs = list(map(json.loads, file))

with ThreadPoolExecutor() as executor:
    batches = executor.map(client.batches.retrieve, [log["id"] for log in logs])
    counts = [batch.request_counts for batch in tqdm(batches, total=len(logs))]

completed = sum(count.completed for count in counts)
failed = sum(count.failed for count in counts)
total = sum(count.total for count in counts)

print("done", completed, sep="\t")
print("failed", failed, sep="\t")
print("total", total, sep="\t")
print(f"{round(100 * completed / total, 2)}% finished")
