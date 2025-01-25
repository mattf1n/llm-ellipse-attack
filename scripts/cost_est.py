import numpy as np
import pandas as pd

df = (
    pd.DataFrame(
        dict(
            model=[
                "pythia-70m",
                "babbage-002",
                "gpt-3.5-turbo",
                "gpt-4o",
                "gpt-4",
                "gpt-4-turbo",
                "llama-3-70b",
                "bloom",
            ],
            embed_size=[512, 1536, 4650, None, None, None, 8192, 14336],
            vocab_size=[
                50304,
                101281,
                101281,
                200_000,
                101_281,
                117_732,
                128256,
                250880,
            ],
            input_cost_per_mil=[None, 0.40, 0.50, 5.00, 30.00, 10.00, None, None],
            output_cost_per_mil=[None, 0.40, 1.50, 15.00, 60.00, 30.00, None, None],
            chat_model=[None, False, True, True, True, True, False, False],
        )
    )
    .astype(dict(embed_size="Int64", vocab_size="Int64"))
    .assign(input_cost=lambda df: df.input_cost_per_mil / 1_000_000)
    .assign(output_cost=lambda df: df.output_cost_per_mil / 1_000_000)
    .assign(
        samples_required=lambda df: (
            pow(df.embed_size - 1, 2) + 3 * (df.embed_size - 1)
        )
        // 2
    )
    .assign(
        avg_prefix_length=lambda df: (
            np.ceil(np.log(df.samples_required) / np.log(df.vocab_size))
            * (df.samples_required - df.vocab_size)
            + df.vocab_size
        )
        / df.samples_required
    )
    .assign(
        query_cost=lambda df: df.input_cost * (df.avg_prefix_length + 7 * df.chat_model)
        + df.output_cost
    )
    .assign(
        image_query_cost=lambda df: df.input_cost * (1 + 7 * df.chat_model)
        + df.output_cost
    )
    .assign(
        image_sample_cost=lambda df: df.image_query_cost / 2 * (df.embed_size + 1000)
    )
    .assign(sample_cost=lambda df: df.query_cost / 2 * (df.embed_size - 1))
    .assign(image_cost=lambda df: df.image_sample_cost * (df.embed_size + 1000))
    .assign(total_cost=lambda df: df.sample_cost * df.samples_required)
    .astype(dict(total_cost="Float64"))[
        [
            "model",
            "embed_size",
            "image_cost",
            "vocab_size",
            "samples_required",
            "total_cost",
        ]
    ][[True, True, True, True, False, False, True, True]]
)
print(df)

(
    df.style.relabel_index(
        [
            "Model",
            "Embed size",
            r"Image (\$)",
            "Vocab size",
            "Samples",
            r"Ellipse (\$)",
        ],
        axis="columns",
    )
    .hide(axis="index")
    .format(na_rep="{--}")
    .format(subset="image_cost", precision=2, na_rep="{--}")
    .format(subset="total_cost", precision=0, na_rep="{--}")
    .format(subset=pd.IndexSlice[1, "embed_size"], formatter=r"{}\tnote{{a}}")
    .format(subset=pd.IndexSlice[2, "embed_size"], formatter=r"{}\tnote{{b}}")
    .format(subset=pd.IndexSlice[3:4, :], na_rep=r"{?}")
    .format(subset="model", formatter=r"\texttt{{{}}}")
    .to_latex(
        "tab/models.tex",
        siunitx=True,
        hrules=True,
        column_format=(
            "@{}"
            "l"
            "S[table-format=5]"
            "S[table-format=2.2]"
            "S[table-format=6]"
            "S[table-format=9]"
            "S[table-format=6]@{}"
        ),
    )
)
