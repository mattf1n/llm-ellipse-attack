import pandas as pd

df = pd.read_pickle("data/error_data.pkl").set_index(("fnames", None)).set_index(("Samples", None), append=True)

df1 = (
        df.stack(1, future_stack=True)
        .reorder_levels([2,0,1])
        .sort_index(level=0, sort_remaining=False)
        .reset_index(level=1)
        .rename(columns={("fnames", None): "fnames"})
        )

print(df1)

breakpoint()
