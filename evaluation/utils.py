import re

def extract_characters_regex(s):
    s = s.strip()
    
    # Use regex to match A, B, C, D that are not adjacent to other letters
    # (?<![a-zA-Z]) ensures no letter before
    # (?![a-zA-Z]) ensures no letter after
    pattern = r'(?<![a-zA-Z])[ABCD](?![a-zA-Z])'
    matches = re.findall(pattern, s)
    
    if matches:
        return matches[-1]  # Return the last match
    
    return ""

def handle_arg_string(v):
    """Convert string value to int, float, bool or string."""
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    
    try:
        return int(v)
    except ValueError:
        pass
    
    try:
        return float(v)
    except ValueError:
        pass
        
    return v

def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]}
    return args_dict

def parse_model_args(model_args=None, additional_config=None):
    """
    Parse model arguments into a unified dictionary format.
    
    Args:
        model_args: String arguments in format "key1=val1,key2=val2" or dict of arguments
        additional_config: Additional configuration parameters to merge with model_args
        
    Returns:
        dict: Combined dictionary of all model arguments
    """
    # Initialize empty dict if model_args is None
    if model_args is None:
        model_args = {}
    
    # Convert string args to dict if needed
    if isinstance(model_args, str):
        model_args = simple_parse_args_string(model_args)
    elif not isinstance(model_args, dict):
        raise ValueError(f"model_args must be string or dict, got {type(model_args)}")
        
    # Initialize empty dict if additional_config is None
    additional_config = {} if additional_config is None else additional_config
    
    # Merge the two dictionaries, additional_config takes precedence
    return {**model_args, **additional_config}