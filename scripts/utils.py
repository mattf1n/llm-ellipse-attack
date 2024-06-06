def get_second_order_terms(x):
    sample_size, hidden_size = x.shape
    outer_prod = x.reshape(sample_size, hidden_size, 1) * x.reshape(
        sample_size, 1, hidden_size
    )
    return (outer_prod).reshape(sample_size, hidden_size * hidden_size)
