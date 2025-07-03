"""
Unit tests for data_manager.py

Tests the DataManager class functionality including data serialization,
deserialization, format detection, and temporary file management.
"""

import os
import json
import csv
import tempfile
import pytest
from unittest.mock import Mock, patch, mock_open

from data_manager import DataManager, DataFile


class TestDataFile:
  """Test cases for the DataFile dataclass"""

  def test_data_file_initialization(self) -> None:
    """Test DataFile with required parameters"""
    data_file = DataFile(
      path='/tmp/test.csv',
      format='csv',
      description='Test CSV file'
    )

    assert data_file.path == '/tmp/test.csv'
    assert data_file.format == 'csv'
    assert data_file.description == 'Test CSV file'
    assert data_file.cleanup_needed is True
    assert data_file.created_at > 0

  def test_data_file_custom_cleanup(self) -> None:
    """Test DataFile with custom cleanup setting"""
    data_file = DataFile(
      path='/tmp/test.json',
      format='json',
      description='Test JSON file',
      cleanup_needed=False
    )

    assert data_file.cleanup_needed is False


class TestDataManager:
  """Test cases for the DataManager class"""

  @pytest.fixture
  def data_manager(self) -> DataManager:
    """Fixture providing a DataManager instance"""
    return DataManager()

  @pytest.fixture
  def temp_dir(self):
    """Fixture providing a temporary directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
      yield temp_dir

  def test_initialization(self, data_manager: DataManager) -> None:
    """Test DataManager initialization"""
    assert data_manager.temp_dir is not None
    assert data_manager.created_files == []
    assert 'csv' in data_manager.supported_formats
    assert 'json' in data_manager.supported_formats
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

  def test_infer_format_file_paths(self, data_manager: DataManager, temp_dir: str) -> None:
    """Test format inference for file paths"""
    # Create temporary files to test path inference
    csv_file = os.path.join(temp_dir, 'test.csv')
    json_file = os.path.join(temp_dir, 'test.json')
    txt_file = os.path.join(temp_dir, 'test.txt')

    with open(csv_file, 'w') as f:
      f.write('header\nvalue')
    with open(json_file, 'w') as f:
      f.write('{"key": "value"}')
    with open(txt_file, 'w') as f:
      f.write('text content')

    assert data_manager.infer_format(csv_file) == 'csv'
    assert data_manager.infer_format(json_file) == 'json'
    assert data_manager.infer_format(txt_file) == 'text'

  def test_serialize_list_as_json(self, data_manager: DataManager) -> None:
    """Test serializing a list as JSON"""
    test_data = [1, 2, 3, 4]

    data_file = data_manager.serialize_input(test_data, format='json')

    assert data_file.format == 'json'
    assert data_file.path.endswith('.json')
    assert os.path.exists(data_file.path)
    assert 'List with 4 items' in data_file.description

    # Verify content
    with open(data_file.path, 'r') as f:
      loaded_data = json.load(f)
    assert loaded_data == test_data

  def test_serialize_dict_as_json(self, data_manager: DataManager) -> None:
    """Test serializing a dictionary as JSON"""
    test_data = {'name': 'John', 'age': 30, 'city': 'NYC'}

    data_file = data_manager.serialize_input(test_data, format='json')

    assert data_file.format == 'json'
    assert 'Dictionary with 3 keys' in data_file.description
    assert 'name' in data_file.description

    # Verify content
    with open(data_file.path, 'r') as f:
      loaded_data = json.load(f)
    assert loaded_data == test_data

  def test_serialize_list_as_csv(self, data_manager: DataManager) -> None:
    """Test serializing a list as CSV"""
    test_data = [1, 2, 3, 4]

    data_file = data_manager.serialize_input(test_data, format='csv')

    assert data_file.format == 'csv'
    assert data_file.path.endswith('.csv')

    # Verify content
    with open(data_file.path, 'r') as f:
      reader = csv.reader(f)
      rows = list(reader)
    assert rows[0] == ['value']  # Header
    assert [row[0] for row in rows[1:]] == ['1', '2', '3', '4']

  def test_serialize_list_of_dicts_as_csv(self, data_manager: DataManager) -> None:
    """Test serializing list of dictionaries as CSV"""
    test_data = [
      {'name': 'Alice', 'age': 25},
      {'name': 'Bob', 'age': 30}
    ]

    data_file = data_manager.serialize_input(test_data, format='csv')

    # Verify content
    with open(data_file.path, 'r') as f:
      reader = csv.DictReader(f)
      loaded_data = list(reader)

    assert len(loaded_data) == 2
    assert loaded_data[0]['name'] == 'Alice'
    assert loaded_data[1]['age'] == '30'

  def test_serialize_string_as_text(self, data_manager: DataManager) -> None:
    """Test serializing a string as text"""
    test_data = 'Hello, World!'

    data_file = data_manager.serialize_input(test_data, format='text')

    assert data_file.format == 'text'
    assert data_file.path.endswith('.txt')

    # Verify content
    with open(data_file.path, 'r') as f:
      content = f.read()
    assert content == test_data

  def test_serialize_auto_format(self, data_manager: DataManager) -> None:
    """Test automatic format detection during serialization"""
    test_data = {'key': 'value'}

    data_file = data_manager.serialize_input(test_data, format='auto')

    assert data_file.format == 'dict'  # Should be inferred as dict

  def test_serialize_unsupported_format(self, data_manager: DataManager) -> None:
    """Test error handling for unsupported formats"""
    test_data = [1, 2, 3]

    with pytest.raises(ValueError, match='Unsupported format'):
      data_manager.serialize_input(test_data, format='unsupported')

  def test_deserialize_json(self, data_manager: DataManager, temp_dir: str) -> None:
    """Test deserializing JSON data"""
    test_data = {'name': 'Alice', 'values': [1, 2, 3]}
    json_file = os.path.join(temp_dir, 'test.json')

    with open(json_file, 'w') as f:
      json.dump(test_data, f)

    result = data_manager.deserialize_output(json_file, 'json')
    assert result == test_data

  def test_deserialize_csv(self, data_manager: DataManager, temp_dir: str) -> None:
    """Test deserializing CSV data"""
    csv_file = os.path.join(temp_dir, 'test.csv')

    with open(csv_file, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['name', 'age'])
      writer.writerow(['Alice', '25'])
      writer.writerow(['Bob', '30'])

    result = data_manager.deserialize_output(csv_file, 'csv')
    assert len(result) == 2
    assert result[0]['name'] == 'Alice'

  def test_deserialize_text(self, data_manager: DataManager, temp_dir: str) -> None:
    """Test deserializing text data"""
    test_content = 'Hello, World!'
    text_file = os.path.join(temp_dir, 'test.txt')

    with open(text_file, 'w') as f:
      f.write(test_content)

    result = data_manager.deserialize_output(text_file, 'text')
    assert result == test_content

  def test_deserialize_auto_format(self, data_manager: DataManager, temp_dir: str) -> None:
    """Test automatic format detection during deserialization"""
    json_file = os.path.join(temp_dir, 'test.json')
    test_data = {'key': 'value'}

    with open(json_file, 'w') as f:
      json.dump(test_data, f)

    result = data_manager.deserialize_output(json_file, 'auto')
    assert result == test_data

  def test_deserialize_missing_file(self, data_manager: DataManager) -> None:
    """Test error handling for missing files"""
    with pytest.raises(FileNotFoundError):
      data_manager.deserialize_output('/nonexistent/file.json', 'json')

  def test_cleanup(self, data_manager: DataManager) -> None:
    """Test cleanup of temporary files"""
    # Create some test data files
    test_data1 = [1, 2, 3]
    test_data2 = {'key': 'value'}

    data_file1 = data_manager.serialize_input(test_data1)
    data_file2 = data_manager.serialize_input(test_data2)

    # Verify files exist
    assert os.path.exists(data_file1.path)
    assert os.path.exists(data_file2.path)
    assert len(data_manager.created_files) == 2

    # Cleanup
    cleaned_count = data_manager.cleanup()

    # Verify cleanup
    assert cleaned_count == 2
    assert not os.path.exists(data_file1.path)
    assert not os.path.exists(data_file2.path)
    assert len(data_manager.created_files) == 0

  def test_get_data_summary(self, data_manager: DataManager) -> None:
    """Test data summary generation"""
    # Create some test files
    data_file1 = data_manager.serialize_input([1, 2, 3], format='list')
    data_file2 = data_manager.serialize_input({'key': 'value'}, format='dict')

    summary = data_manager.get_data_summary()

    assert summary['total_files'] == 2
    assert 'list' in summary['formats']
    assert 'dict' in summary['formats']
    assert summary['total_size_bytes'] > 0
    assert summary['oldest_file_age'] >= 0

  def test_generate_data_description(self, data_manager: DataManager) -> None:
    """Test data description generation"""
    # Test list description
    list_desc = data_manager._generate_data_description([1, 2, 3], 'list')
    assert 'List with 3 items' in list_desc
    assert 'Items are of type: int' in list_desc
    assert 'Format: list' in list_desc

    # Test dict description
    dict_data = {'name': 'Alice', 'age': 30}
    dict_desc = data_manager._generate_data_description(dict_data, 'dict')
    assert 'Dictionary with 2 keys' in dict_desc
    assert 'name' in dict_desc
    assert 'Format: dict' in dict_desc

  def test_file_tracking(self, data_manager: DataManager) -> None:
    """Test that created files are properly tracked"""
    initial_count = len(data_manager.created_files)

    data_file = data_manager.serialize_input([1, 2, 3])

    assert len(data_manager.created_files) == initial_count + 1
    assert data_file in data_manager.created_files

  @patch('data_manager.PANDAS_AVAILABLE', False)
  def test_pandas_not_available(self, data_manager: DataManager) -> None:
    """Test behavior when pandas is not available"""
    # This test simulates pandas not being available
    # Should handle gracefully when trying to work with DataFrames
    pass  # Placeholder for pandas-specific tests

  @patch('data_manager.NUMPY_AVAILABLE', False)
  def test_numpy_not_available(self, data_manager: DataManager) -> None:
    """Test behavior when numpy is not available"""
    # This test simulates numpy not being available
    # Should handle gracefully when trying to work with arrays
    pass  # Placeholder for numpy-specific tests


if __name__ == '__main__':
  pytest.main([__file__, '-v'])