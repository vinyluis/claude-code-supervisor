"""
Data Manager for Claude Code Supervisor

Handles serialization, deserialization, and management of various data formats
for input/output operations with Claude Code.

Supports: lists, dicts, pandas DataFrames, numpy arrays, CSV files, JSON files,
plain text, and automatic format detection.
"""

import os
import json
import csv
import tempfile
import pickle
import shutil
from pathlib import Path
from typing import Any, Union, Optional, Dict, List, Tuple
from dataclasses import dataclass
import time

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class DataFile:
    """Represents a temporary data file created for Claude Code processing"""
    path: str
    format: str
    description: str
    cleanup_needed: bool = True
    created_at: float = 0.0

    def __post_init__(self) -> None:
        self.created_at = time.time()


class DataManager:
    """
    Manages data serialization, deserialization, and temporary file operations
    for Claude Code Supervisor I/O operations.
    
    Features:
    - Automatic format detection for various data types
    - Temporary file creation and cleanup
    - Support for multiple data formats (CSV, JSON, lists, dicts, etc.)
    - Data validation and type conversion
    - Memory-efficient handling of large datasets
    """
    
    def __init__(self, temp_dir: str | None = None) -> None:
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.created_files: List[DataFile] = []
        self.supported_formats = {
            'csv', 'json', 'list', 'dict', 'array', 'string', 'text', 
            'dataframe', 'pickle', 'auto'
        }
    
    def infer_format(self, data: Any) -> str:
        """
        Automatically infer the data format from the input data type.
        
        Args:
            data: Input data of any supported type
            
        Returns:
            String representing the inferred format
        """
        if isinstance(data, str):
            # Check if it's a file path
            if os.path.exists(data):
                ext = Path(data).suffix.lower()
                if ext == '.csv':
                    return 'csv'
                elif ext in ['.json', '.jsonl']:
                    return 'json'
                elif ext in ['.txt', '.md']:
                    return 'text'
                elif ext in ['.pkl', '.pickle']:
                    return 'pickle'
            return 'string'
        elif isinstance(data, list):
            return 'list'
        elif isinstance(data, dict):
            return 'dict'
        elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return 'dataframe'
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            return 'array'
        else:
            return 'pickle'  # Default fallback for complex objects
    
    def serialize_input(self, data: Any, format: str = 'auto', 
                       name_prefix: str = 'input') -> DataFile:
        """
        Serialize input data to a temporary file that Claude Code can access.
        
        Args:
            data: Input data to serialize
            format: Target format ('auto' for automatic detection)
            name_prefix: Prefix for the temporary file name
            
        Returns:
            DataFile object with path and metadata
            
        Raises:
            ValueError: If format is unsupported or data cannot be serialized
        """
        if format == 'auto':
            format = self.infer_format(data)
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. "
                           f"Supported: {', '.join(self.supported_formats)}")
        
        # Create temporary file with appropriate extension
        file_extensions = {
            'csv': '.csv',
            'json': '.json', 
            'list': '.json',
            'dict': '.json',
            'array': '.csv',
            'dataframe': '.csv',
            'string': '.txt',
            'text': '.txt',
            'pickle': '.pkl'
        }
        
        ext = file_extensions.get(format, '.txt')
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=ext, 
            prefix=f'{name_prefix}_',
            dir=self.temp_dir,
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            self._write_data_to_file(data, temp_path, format)
            
            # Create description for Claude Code
            description = self._generate_data_description(data, format)
            
            data_file = DataFile(
                path=temp_path,
                format=format,
                description=description
            )
            self.created_files.append(data_file)
            return data_file
            
        except Exception as e:
            # Clean up on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise ValueError(f"Failed to serialize data: {e}")
    
    def _write_data_to_file(self, data: Any, file_path: str, format: str) -> None:
        """Write data to file based on format"""
        if format == 'csv':
            self._write_csv(data, file_path)
        elif format in ['json', 'list', 'dict']:
            self._write_json(data, file_path)
        elif format in ['string', 'text']:
            self._write_text(data, file_path)
        elif format == 'dataframe':
            self._write_dataframe_csv(data, file_path)
        elif format == 'array':
            self._write_array_csv(data, file_path)
        elif format == 'pickle':
            self._write_pickle(data, file_path)
        else:
            raise ValueError(f"Unknown format for writing: {format}")
    
    def _write_csv(self, data: Any, file_path: str) -> None:
        """Write data as CSV"""
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # List of dictionaries -> CSV with headers
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
                # List of lists/tuples -> CSV without headers
                writer = csv.writer(f)
                writer.writerows(data)
            elif isinstance(data, list):
                # Simple list -> single column CSV
                writer = csv.writer(f)
                writer.writerow(['value'])  # Header
                for item in data:
                    writer.writerow([item])
            else:
                raise ValueError(f"Cannot write {type(data)} as CSV")
    
    def _write_json(self, data: Any, file_path: str) -> None:
        """Write data as JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _write_text(self, data: Any, file_path: str) -> None:
        """Write data as plain text"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(data))
    
    def _write_dataframe_csv(self, data: Any, file_path: str) -> None:
        """Write pandas DataFrame as CSV"""
        if not PANDAS_AVAILABLE:
            raise ValueError("Pandas not available for DataFrame operations")
        data.to_csv(file_path, index=False)
    
    def _write_array_csv(self, data: Any, file_path: str) -> None:
        """Write numpy array as CSV"""
        if not NUMPY_AVAILABLE:
            raise ValueError("NumPy not available for array operations")
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if data.ndim == 1:
                # 1D array -> single column
                writer.writerow(['value'])
                for item in data:
                    writer.writerow([item])
            elif data.ndim == 2:
                # 2D array -> multiple columns
                for row in data:
                    writer.writerow(row)
            else:
                raise ValueError(f"Cannot write {data.ndim}D array as CSV")
    
    def _write_pickle(self, data: Any, file_path: str) -> None:
        """Write data using pickle"""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _generate_data_description(self, data: Any, format: str) -> str:
        """Generate human-readable description of the data for Claude Code"""
        descriptions = []
        
        if isinstance(data, list):
            descriptions.append(f"List with {len(data)} items")
            if data:
                first_type = type(data[0]).__name__
                descriptions.append(f"Items are of type: {first_type}")
        elif isinstance(data, dict):
            descriptions.append(f"Dictionary with {len(data)} keys")
            descriptions.append(f"Keys: {list(data.keys())[:5]}{'...' if len(data) > 5 else ''}")
        elif isinstance(data, str):
            descriptions.append(f"String with {len(data)} characters")
        elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            descriptions.append(f"DataFrame with {len(data)} rows and {len(data.columns)} columns")
            descriptions.append(f"Columns: {list(data.columns)}")
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            descriptions.append(f"NumPy array with shape {data.shape}")
            descriptions.append(f"Data type: {data.dtype}")
        
        descriptions.append(f"Format: {format}")
        return "; ".join(descriptions)
    
    def deserialize_output(self, file_path: str, expected_format: str = 'auto') -> Any:
        """
        Deserialize output data from a file created by Claude Code.
        
        Args:
            file_path: Path to the output file
            expected_format: Expected format of the output data
            
        Returns:
            Deserialized Python object
            
        Raises:
            FileNotFoundError: If output file doesn't exist
            ValueError: If format is unsupported or file cannot be parsed
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Output file not found: {file_path}")
        
        if expected_format == 'auto':
            expected_format = self._detect_file_format(file_path)
        
        try:
            if expected_format == 'csv':
                return self._read_csv(file_path)
            elif expected_format in ['json', 'list', 'dict']:
                return self._read_json(file_path)
            elif expected_format in ['string', 'text']:
                return self._read_text(file_path)
            elif expected_format == 'pickle':
                return self._read_pickle(file_path)
            else:
                # Default to text for unknown formats
                return self._read_text(file_path)
                
        except Exception as e:
            raise ValueError(f"Failed to deserialize output from {file_path}: {e}")
    
    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format from extension and content"""
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            return 'csv'
        elif ext in ['.json', '.jsonl']:
            return 'json'
        elif ext in ['.pkl', '.pickle']:
            return 'pickle'
        else:
            return 'text'
    
    def _read_csv(self, file_path: str) -> Union[List[Dict], List[List]]:
        """Read CSV file as list of dictionaries or list of lists"""
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to detect if first row is header
            sample = f.read(1024)
            f.seek(0)
            
            sniffer = csv.Sniffer()
            try:
                has_header = sniffer.has_header(sample)
                reader = csv.DictReader(f) if has_header else csv.reader(f)
                return list(reader)
            except:
                # Fallback to basic reader
                reader = csv.reader(f)
                return list(reader)
    
    def _read_json(self, file_path: str) -> Any:
        """Read JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _read_text(self, file_path: str) -> str:
        """Read text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _read_pickle(self, file_path: str) -> Any:
        """Read pickle file"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def cleanup(self) -> None:
        """Clean up all temporary files created by this DataManager"""
        cleaned_count = 0
        for data_file in self.created_files:
            if data_file.cleanup_needed and os.path.exists(data_file.path):
                try:
                    os.unlink(data_file.path)
                    cleaned_count += 1
                except OSError:
                    pass  # File already deleted or permission issue
        
        self.created_files.clear()
        return cleaned_count
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all managed data files"""
        return {
            'total_files': len(self.created_files),
            'formats': [df.format for df in self.created_files],
            'total_size_bytes': sum(
                os.path.getsize(df.path) if os.path.exists(df.path) else 0 
                for df in self.created_files
            ),
            'oldest_file_age': min(
                (time.time() - df.created_at for df in self.created_files), 
                default=0
            )
        }
    
    def __del__(self) -> None:
        """Ensure cleanup on object destruction"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor