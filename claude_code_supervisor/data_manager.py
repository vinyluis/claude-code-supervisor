"""
Data Manager for Claude Code Supervisor

Handles in-memory data processing and format detection for input/output operations.
Simplified version that focuses on essential functionality only.
"""

from typing import Any
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np


@dataclass
class DataInfo:
  """Information about processed data"""
  format: str
  description: str
  size: int


class DataManager:
  """
  Manages in-memory data processing for Claude Code Supervisor I/O operations.

  Features:
  - Automatic format detection for common data types
  - Memory-only operations (no file I/O)
  - Format descriptions for Claude Code context
  """

  def __init__(self) -> None:
    self.supported_formats = {'list', 'dict', 'string', 'auto'}
    self.processed_data: list[DataInfo] = []

  def infer_format(self, data: Any) -> str:
    """
    Automatically infer the data format from the input data type.

    Args:
      data: Input data of any supported type

    Returns:
      String representing the inferred format
    """
    if isinstance(data, str):
      return 'string'
    elif isinstance(data, list):
      return 'list'
    elif isinstance(data, dict):
      return 'dict'
    elif isinstance(data, pd.DataFrame):
      return 'dataframe'
    elif isinstance(data, np.ndarray):
      return 'array'
    else:
      return 'dict'

  def get_data_description(self, data: Any, format: str = 'auto') -> str:
    """
    Generate human-readable description of the data for Claude Code context.

    Args:
      data: Input data to describe
      format: Data format ('auto' for automatic detection)

    Returns:
      Human-readable description string
    """
    if format == 'auto':
      format = self.infer_format(data)

    descriptions = []

    if isinstance(data, list):
      descriptions.append(f'List with {len(data)} items')
      if data:
        first_type = type(data[0]).__name__
        descriptions.append(f'Items are of type: {first_type}')
        if isinstance(data[0], dict) and len(data) <= 5:
          descriptions.append(f'Sample item: {data[0]}')
        elif len(data) <= 10:
          descriptions.append(f'Sample items: {data[:3]}{'...' if len(data) > 3 else ''}')

    elif isinstance(data, dict):
      descriptions.append(f'Dictionary with {len(data)} keys')
      key_list = list(data.keys())[:5]
      suffix = '...' if len(data) > 5 else ''
      descriptions.append(f'Keys: {key_list}{suffix}')
      if len(data) <= 5:
        descriptions.append(f'Sample: {dict(list(data.items())[:2])}')

    elif isinstance(data, str):
      descriptions.append(f'String with {len(data)} characters')
      if len(data) <= 100:
        descriptions.append(f'Content: \'{data}\'')
      else:
        descriptions.append(f'Preview: \'{data[:50]}...\'')

    elif isinstance(data, pd.DataFrame):
      descriptions.append(f'DataFrame with {len(data)} rows and {len(data.columns)} columns')
      descriptions.append(f'Columns: {list(data.columns)}')
      if not data.empty:
        descriptions.append(f'Sample row: {data.iloc[0].to_dict()}')

    elif isinstance(data, np.ndarray):
      descriptions.append(f'NumPy array with shape {data.shape}')
      descriptions.append(f'Data type: {data.dtype}')
      if data.size <= 10:
        descriptions.append(f'Values: {data.tolist()}')

    descriptions.append(f'Format: {format}')
    return '; '.join(descriptions)

  def serialize_for_context(self, data: Any, format: str = 'auto',
                            max_items: int = 5) -> str:
    """
    Serialize data to a string representation for Claude Code context.

    Args:
      data: Data to serialize
      format: Data format hint
      max_items: Maximum number of items to include in examples

    Returns:
      String representation suitable for Claude Code prompts
    """
    if format == 'auto':
      format = self.infer_format(data)

    try:
      if isinstance(data, list):
        if len(data) <= max_items:
          return f'Input data (list): {data}'
        else:
          return f'Input data (list with {len(data)} items): {data[:max_items]}... (showing first {max_items})'

      elif isinstance(data, dict):
        if len(data) <= max_items:
          return f'Input data (dict): {data}'
        else:
          items = dict(list(data.items())[:max_items])
          return f'Input data (dict with {len(data)} keys): {items}... (showing first {max_items})'

      elif isinstance(data, str):
        if len(data) <= 200:
          return f'Input data (string): \'{data}\''
        else:
          return f'Input data (string, {len(data)} chars): \'{data[:100]}...\''

      elif isinstance(data, pd.DataFrame):
        return f'Input data (DataFrame {data.shape}): {data.head(max_items).to_dict('records')}'

      elif isinstance(data, np.ndarray):
        if data.size <= max_items * 2:
          return f'Input data (array {data.shape}): {data.tolist()}'
        else:
          return f'Input data (array {data.shape}): {data.flat[:max_items].tolist()}... (showing first {max_items})'

      else:
        # Fallback: try to JSON serialize
        return f'Input data ({format}): {json.dumps(data, default=str)[:200]}'

    except Exception:
      return f'Input data ({format}): <{type(data).__name__} object>'

  def record_operation(self, data: Any, format: str, operation: str = 'process') -> None:
    """Record a data processing operation for tracking"""
    description = self.get_data_description(data, format)

    # Estimate size
    try:
      if isinstance(data, (list, dict)):
        size = len(str(data))
      elif isinstance(data, str):
        size = len(data)
      elif isinstance(data, pd.DataFrame):
        size = data.memory_usage(deep=True).sum()
      elif isinstance(data, np.ndarray):
        size = data.nbytes
      else:
        size = len(str(data))
    except:
      size = 0

    info = DataInfo(
      format=format,
      description=description,
      size=size
    )
    self.processed_data.append(info)

  def get_summary(self) -> dict[str, Any]:
    """Get summary of data processing operations"""
    return {
      'total_operations': len(self.processed_data),
      'formats_processed': [info.format for info in self.processed_data],
      'total_data_size': sum(info.size for info in self.processed_data)
    }
