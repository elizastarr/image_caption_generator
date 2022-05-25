from pathlib import Path
from config.core import (
    create_config,
    fetch_config_from_yaml,
)

import pytest
from pydantic import ValidationError


def test_missing_config_field_raises_validation_error(tmpdir):
    # Given
    configs_dir = Path(tmpdir)  # pytest built-in tmpdir fixture
    config_test = configs_dir / "sample_config.yml"
    TEST_CONFIG_TEXT = """test_val_size: 1000"""
    config_test.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_test)

    # When
    with pytest.raises(ValidationError) as error:
        create_config(parsed_config=parsed_config)

    # Then
    print(error)
    assert "field required" in str(error.value)
