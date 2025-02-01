import numpy as np
import ellipse_attack.representations as rep

def assert_rotation(a, b):
    soln = np.linalg.solve(a, b)
    np.testing.assert_allclose(soln.T @ soln, np.eye(soln.shape))
    np.testing.assert_allclose(np.linalg.det(soln), 1.0)

def test_convert():
    hidden_size = 64
    prenorm = np.arange(hidden_size) 
    centered = prenorm - prenorm.mean()
    model = rep.Model(
            gamma=np.arange(hidden_size),
            beta=np.arange(hidden_size),
            W=np.arange(hidden_size),)
    ellipse = rep.ellipse_of_model(**model_params)
    model_recovered = rep.model_of_ellipse(**model_params)
    model_fwd = rep.apply_model(prenorm, **model_params)
    
    ellipse_fwd = rep.apply_ellipse(prenorm, **ellipse_params)
    np.testing.assert_allclose(model_fwd
