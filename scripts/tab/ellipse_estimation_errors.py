import pandas as pd

df = (
    pd.read_pickle("data/error_data.pkl")
    .set_index(("fnames", None))
    .set_index(("Samples", None), append=True)
)

# Sample size
# Columns: Rot Angle, Bias RMS, Bias max diff, Stretch max
df_ = (
    df.drop(
        columns=[
            col for col in df.columns if "rot2" in col and "Angle" not in col
        ]
    )[
        [
            ("Angle", "rot2"),
            ("RMS", "bias"),
            ("Max diff.", "bias"),
            ("RMS", "stretch"),
            ("Max rel. diff.", "stretch"),
        ]
    ]
    .stack(1, future_stack=True)
    .reset_index(level=0)
    .rename(columns={("fnames", None): "fnames"})
    .rename_axis(index=["samples", "param"])
    .reset_index(level=0)
    .assign(samples=lambda df: df.samples.astype(int))
    .loc[lambda df: ~df.fnames.str.contains("pred")]
    .drop(columns="fnames")
    .set_index("samples", append=True)
    .unstack(0)
    .reorder_levels([1, 0], axis="columns")
    .T.reindex(index=["rot2", "stretch", "bias"], level=0)
    .T.drop(
        columns=[
            ("bias", "Angle"),
            ("bias", "Max rel. diff."),
            ("stretch", "Angle"),
            ("stretch", "Max diff."),
            ("rot2", "RMS"),
            ("rot2", "Max rel. diff."),
            ("rot2", "Max diff."),
        ]
    )
    .rename_axis(columns=("param", "metric"))
)

print(df_)
