    # Get and save model params
    W = model.lm_head.weight.cpu().numpy().T
    gamma = model.transformer.ln_f.weight.cpu().numpy()
    beta = model.transformer.ln_f.bias.cpu().numpy()
    final_layer = Model(stretch=gamma, bias=beta, unembed=W)
    os.makedirs("data/model", exist_ok=True)
    np.savez(f"data/model/{os.path.basename(model_name)}.npz", **asdict(final_layer))
