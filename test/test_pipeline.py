import os
import json
import tempfile
import pytest
import pandas as pd
from utils import import_attributes

@pytest.fixture
def sample_attributes_json():
    """Creates a temporary attributes.json file with valid content."""
    data = [
        {
            "id": "1",
            "type": "top",
            "color1": "red",
            "color2": "none",
            "pattern": "solid",
            "dress_code": "casual",
            "material": "cotton",
            "seasonality": "summer",
            "fit": "slim"
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "attributes.json")
        with open(file_path, "w") as f:
            json.dump(data, f)
        yield tmpdir  # return the directory path containing attributes.json

def test_import_attributes_success(sample_attributes_json):
    """Test successful parsing and encoding of attributes.json."""
    df = import_attributes(sample_attributes_json)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert all(col in df.columns for col in ["type", "color1", "pattern", "material", "fit"])
    assert df["type"].iloc[0] == 0  # "top" => 0
