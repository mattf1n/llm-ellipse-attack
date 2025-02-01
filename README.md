Description of files

- `scripts/ying.py`: Fits ellipse to random weights via Ying's method and CVXPY.
- `scripts/get_ellipse.py`: functions for getting the ellipse using CVXPY given points on the ellipse. Uses Ying's method.
- `scripts/sample_from_model.py`: does batch inference and saves the results, then times the solver for different numbers of samples, saving the resulting parameters.
- `scripts/validate_ellipse_pred.py`: Validates the predicted ellipse parameters.
- `scripts/utils.py`: some general utils
- `scripts/cost_est.py`: estimating the cost of inference to obtain the ellipse for various models.
- `scripts/centering.py`: shows what subtracting the mean does.
- `scripts/openai_batch_inference/`: scripts for batch inference with OpenAI.
- `scripts/lstsq_fitting`: loads samples from model and computes ellipse using `torch.linalg.solve`. Deprecated.
- `scripts/nanoGPT`: scripts for training a tiny LM.

Workflow: extracting the ellipse of a real LM.
```sh
# Generate the samples
python scripts/sample_from_model.py

# Filter for low-variance norm outputs.
python scripts/save_hidden_size_centered_norms.py 

# Solve for the ellipse with various numbers of outputs, 
# save the solve times.
# The `--filter` flag uses the low-variance outputs
python scripts/time_ellipse_solving.py [--filter]

# Evaluate the goodness of fit
# The `--filter` flag uses the low-variance outputs
python scripts/validate_ellipse_pred.py [--filter]
```
