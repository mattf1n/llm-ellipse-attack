import json, sys, itertools
import tiktoken


def main():
    model = "gpt-3.5-turbo-0125"
    enc = tiktoken.encoding_for_model(model)
    print("Decoding batch", file=sys.stderr)
    decoded_tokens: set[bytes] = set(
        enc.decode_batch([tok] for tok in range(enc.n_vocab - 1000))
    )
    print("Done decoding batch", file=sys.stderr)
    # Ensure unique strings decode back to single tokens
    tokens = filter(lambda tok: len(enc.encode(tok)) == 1, decoded_tokens)
    dim_bound = 4700
    messages_batch: list[dict[str, str]] = (
        dict(content=token, role="user") for token in tokens
    )
    queries = (
        dict(
            logit_bias={str(tok): 100},
            logprobs=True,
            messages=[messages],
            model=model,
            max_tokens=1,
            temperature=0,
            seed=0,
        )
        for messages in itertools.islice(messages_batch, dim_bound)
        for tok in range(dim_bound)
    )
    requests = (
        dict(custom_id=str(i), method="POST", url="/v1/chat/completions", body=body)
        for i, body in enumerate(queries)
    )
    max_queries_per_batch = 50_000
    total_queries = pow(dim_bound, 2)
    for i, batch in enumerate(itertools.batched(requests, max_queries_per_batch)):
        print(
            f"Writing batch {i} of {total_queries // max_queries_per_batch}",
            file=sys.stderr,
        )
        with open(f"data/queries/{i}.jsonl", "w") as file:
            for request in batch:
                print(json.dumps(request), file=file)


if __name__ == "__main__":
    main()
