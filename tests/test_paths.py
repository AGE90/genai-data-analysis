from genaianalysis.utils.paths import data_raw_dir, project_dir


def test_paths_resolve_from_project_root():
    assert (project_dir() / "pyproject.toml").exists()
    assert data_raw_dir("x.txt") == project_dir("data", "raw", "x.txt")
