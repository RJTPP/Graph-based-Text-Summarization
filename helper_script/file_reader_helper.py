import os
from typing import Optional

def read_from_file(path: str, reader_function: Optional[callable] = None, read_logging: bool = False, **kwargs):
    if not os.path.exists(path):
        if read_logging:
            print(f"File {path} does not exist")
        return None
    
    if read_logging:
        print(f"Reading content from {path}")

    if reader_function is None:
        with open(path, "r") as f:
            return f.read()
            
    return reader_function(path, **kwargs)

def write_to_file(path: str, content: str = None, overwrite: bool = False):
    if content is None:
        raise ValueError("No content given")
    
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File {path} already exists")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        f.write(content)
