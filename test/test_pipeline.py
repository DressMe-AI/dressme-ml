import os
import json
import tempfile
import pytest
import pandas as pd
from utils import import_attributes, call_data

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

@pytest.fixture
def sample_encoded_df():
    """Returns a minimal encoded DataFrame with top/bottom IDs and mapped values."""
    return pd.DataFrame([
        {"id": "1", "color1": 0, "pattern": 0, "material": 0, "fit": 4},  # top
        {"id": "2", "color1": 1, "pattern": 2, "material": 1, "fit": 3}   # bottom
    ])

@pytest.fixture
def sample_combinations_file():
    """Creates a temp combination_scored.txt file with one positive and one neutral combo."""
    lines = [
        "top:1,bottom:2,1\n",    # valid
        "top:1,bottom:2,0\n"     # should be ignored
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "combination_scored.txt")
        with open(file_path, "w") as f:
            f.writelines(lines)
        yield tmpdir  # path to dir containing combination_scored.txt

def test_call_data_success(sample_encoded_df, sample_combinations_file):
    """Test call_data returns correct feature matrix and labels."""
    X, y = call_data(sample_encoded_df, sample_combinations_file)
    
    assert X.shape == (1, 8)      # 1 valid sample, 4 features per item × 2 items
    assert y.shape == (1,)
    assert y[0] in (0, 1)         # binary label
