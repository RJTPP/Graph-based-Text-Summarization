import orjson
from typing import Any, List, Union, Optional

def read_json(file_path: str,  encoding: str = "utf-8") -> Any:
    """
    Reads a JSON file and returns its content.
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return orjson.loads(f.read())

def write_json(file_path: str, data: Any, indent: bool = False, encoding: str = "utf-8") -> None:
    with open(file_path, 'w', encoding=encoding) as f:
        if indent:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode(encoding))
        else:
            f.write(orjson.dumps(data).decode(encoding))


def to_json(data: Any, indent: bool = False, encoding: str = "utf-8") -> str:
    if indent:
        return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode(encoding)
    
    return orjson.dumps(data).decode(encoding)


def print_as_json(data: Any, indent: bool = False) -> None:
    if indent:
        print(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8"))
    else:
        print(orjson.dumps(data).decode("utf-8"))


def get_from_nested_key(data: Union[dict, str], keys: List[str], throw_key_error: bool = False, default: Optional[Any] = None) -> Optional[Any]:
    """
    Gets a value from nested dictionaries.

    Args:
        data (Union[dict, str]): The data to retrieve the value from.
        keys (List[str]): The keys to traverse in order.
        default (Optional[Any]): Value to return when the key is not found. Defaults to None.

    Returns:
        Union[Any, None]: The value retrieved from the nested dictionaries.
    """
    if isinstance(data, str):
        data = orjson.loads(data)

    if len(keys) == 0:
        return data

    value = data
    for key in keys:
        value = value.get(key)
        if value is None:
            if throw_key_error:
                raise KeyError(f"Key '{key}' not found in data")
            return default
    return value
