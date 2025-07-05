"""
Unit tests for data_manager.py

Tests the simplified DataManager class functionality including in-memory data processing,
format detection, and context generation.
"""

import pytest

from claude_code_supervisor.data_manager import DataManager, DataInfo


class TestDataInfo:
  """Test cases for the DataInfo dataclass"""

  def test_data_info_initialization(self) -> None:
    """Test DataInfo with required parameters"""
    data_info = DataInfo(
      format='list',
      description='List with 3 items',
      size=24
    )

    assert data_info.format == 'list'
    assert data_info.description == 'List with 3 items'
    assert data_info.size == 24


class TestDataManager:
  """Test cases for the simplified DataManager class"""

  @pytest.fixture
  def data_manager(self) -> DataManager:
    """Fixture providing a DataManager instance"""
    return DataManager()

  def test_initialization(self, data_manager: DataManager) -> None:
    """Test DataManager initialization"""
    assert data_manager.processed_data == []
    assert 'list' in data_manager.supported_formats
    assert 'dict' in data_manager.supported_formats
    assert 'auto' in data_manager.supported_formats

  def test_infer_format_string(self, data_manager: DataManager) -> None:
    """Test format inference for strings"""
    assert data_manager.infer_format('hello world') == 'string'

  def test_infer_format_list(self, data_manager: DataManager) -> None:
    """Test format inference for lists"""
    assert data_manager.infer_format([1, 2, 3]) == 'list'

  def test_infer_format_dict(self, data_manager: DataManager) -> None:
    """Test format inference for dictionaries"""
    assert data_manager.infer_format({'key': 'value'}) == 'dict'

  def test_infer_format_dataframe(self, data_manager: DataManager) -> None:
    """Test format inference for pandas DataFrames"""
    import pandas as pd
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    assert data_manager.infer_format(df) == 'dataframe'

  def test_infer_format_array(self, data_manager: DataManager) -> None:
    """Test format inference for numpy arrays"""
    import numpy as np
    arr = np.array([1, 2, 3])
    assert data_manager.infer_format(arr) == 'array'

  def test_get_data_description_list(self, data_manager: DataManager) -> None:
    """Test data description generation for lists"""
    test_data = [1, 2, 3, 4]
    description = data_manager.get_data_description(test_data)

    assert 'List with 4 items' in description
    assert 'Items are of type: int' in description
    assert 'Format: list' in description

  def test_get_data_description_dict(self, data_manager: DataManager) -> None:
    """Test data description generation for dictionaries"""
    test_data = {'name': 'Alice', 'age': 30, 'city': 'NYC'}
    description = data_manager.get_data_description(test_data)

    assert 'Dictionary with 3 keys' in description
    assert 'name' in description
    assert 'Format: dict' in description

  def test_get_data_description_string(self, data_manager: DataManager) -> None:
    """Test data description generation for strings"""
    test_data = 'Hello, World!'
    description = data_manager.get_data_description(test_data)

    assert 'String with 13 characters' in description
    assert 'Hello, World!' in description
    assert 'Format: string' in description

  def test_serialize_for_context_list(self, data_manager: DataManager) -> None:
    """Test context serialization for lists"""
    test_data = [1, 2, 3, 4]
    result = data_manager.serialize_for_context(test_data)

    assert 'Input data (list):' in result
    assert '[1, 2, 3, 4]' in result

  def test_serialize_for_context_large_list(self, data_manager: DataManager) -> None:
    """Test context serialization for large lists"""
    test_data = list(range(20))
    result = data_manager.serialize_for_context(test_data, max_items=3)

    assert 'list with 20 items' in result
    assert 'showing first 3' in result

  def test_serialize_for_context_dict(self, data_manager: DataManager) -> None:
    """Test context serialization for dictionaries"""
    test_data = {'name': 'Alice', 'age': 30}
    result = data_manager.serialize_for_context(test_data)

    assert 'Input data (dict):' in result
    assert 'Alice' in result

  def test_serialize_for_context_string(self, data_manager: DataManager) -> None:
    """Test context serialization for strings"""
    test_data = 'Hello, World!'
    result = data_manager.serialize_for_context(test_data)

    assert 'Input data (string):' in result
    assert 'Hello, World!' in result

  def test_record_operation(self, data_manager: DataManager) -> None:
    """Test operation recording"""
    test_data = [1, 2, 3]
    initial_count = len(data_manager.processed_data)

    data_manager.record_operation(test_data, 'list', 'process')

    assert len(data_manager.processed_data) == initial_count + 1
    assert data_manager.processed_data[-1].format == 'list'
    assert 'List with 3 items' in data_manager.processed_data[-1].description

  def test_get_summary(self, data_manager: DataManager) -> None:
    """Test summary generation"""
    # Record some operations
    data_manager.record_operation([1, 2, 3], 'list')
    data_manager.record_operation({'key': 'value'}, 'dict')

    summary = data_manager.get_summary()

    assert summary['total_operations'] == 2
    assert 'list' in summary['formats_processed']
    assert 'dict' in summary['formats_processed']
    assert summary['total_data_size'] > 0

  def test_unknown_type_fallback(self, data_manager: DataManager) -> None:
    """Test behavior with unknown data types"""
    # Should fall back to dict format for unknown types
    class MockDataType:
      pass

    mock_obj = MockDataType()
    assert data_manager.infer_format(mock_obj) == 'dict'


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
