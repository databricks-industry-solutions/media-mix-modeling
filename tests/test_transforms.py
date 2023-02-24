import pytest
import os
import sys
import numpy as np
from mediamix.transforms import geometric_adstock_tt, logistic_function
import warnings


def test_geometric_adstock_tt():
    n = 25
    a = 5
    x = np.zeros(n)
    x[a] = 1.0
    y = np.zeros(n)
    y[a:a+10] = np.array([0.5, 0.25, 0.125, 0.063, 0.031, 0.016, 0.008, 0.004, 0.002, 0.001])
    y_hat = geometric_adstock_tt(x, 0.5).eval()
    assert np.allclose(y, y_hat, atol=0.01)


def test_logistic_function():
    pass


if __name__ == '__main__':
    test_geometric_adstock_tt()
    repo_name = 'media-mix-model'

    # Get the path to this notebook, for example "/Workspace/Repos/{username}/{repo-name}".
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

    # Get the repo's root directory name.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
    print(repo_root)

    # Prepare to run pytest from the repo.
    os.chdir(f"/Workspace/{repo_root}/{repo_name}")
    print(os.getcwd())

    # Skip writing pyc files on a readonly filesystem.
    sys.dont_write_bytecode = True

    # Run pytest.
    retcode = pytest.main(["tests/test_transforms.py", "-v", "-p", "no:cacheprovider"])