print(logits.shape)
# logits = logits - np.mean(logits, axis=1, keepdims=True)
rank = np.linalg.matrix_rank(logits, tol=1e-3)
print("Rank is", rank)
n = rank - 1

for samples in [5000, 10_000, 20_000, 30_000, None]:
    start = time.time()
    S_pred, U_pred, bias_pred = get_ellipse(logits[:samples, :n])
    seconds = time.time() - start
    with open("data/times.dat", "a") as times:
        print(samples, seconds, file=times)
    np.savez(
        f"data/ellipse_pred_{samples}_samples.npz",
        S_pred=S_pred,
        U_pred=U_pred,
        bias_pred=bias_pred,
    )
