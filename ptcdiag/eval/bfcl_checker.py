"""Python-only port of the BFCL AST checker.

Adapted from gorilla/berkeley-function-call-leaderboard
(bfcl_eval/eval_checker/ast_eval/ast_checker.py, Apache-2.0). Java/JS branches and
model-specific function-name conversion are dropped because we only evaluate the
Python categories and our prompt keeps the original function names.

The only structural change: the per-parameter body of `simple_function_checker`
is factored out into `check_param`, so the taxonomy can inspect every parameter
instead of stopping at the first failure. `simple_function_checker` still returns
on the first failure, exactly like the original.
"""

import re

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]


def find_description(func_descriptions, name):
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    return func_descriptions


def get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None


def type_checker(param, value, possible_answer, expected_type_description,
                 expected_type_converted, nested_type_converted):
    # Only one level of nested type checking, as in the original.
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}

    is_variable = False
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if possible_answer_type != expected_type_converted:
            is_variable = True

    if type(value) == expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = type_checker(
                            param, value_item, possible_answer_item,
                            str(nested_type_converted), nested_type_converted, None,
                        )
                        if not checker_result["valid"]:
                            flag = False
                            break
                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}

            result["valid"] = False
            result["error"] = [
                f"Nested type checking failed for parameter {param!r}. Expected outer type "
                f"{expected_type_description} with inner type {nested_type_converted}. "
                f"Parameter value: {value!r}."
            ]
            result["error_type"] = "type_error:nested"

    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if type(value) == possible_answer_type:
            result["is_variable"] = True
            return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {param!r}. Expected type {expected_type_description}, "
        f"got {type(value).__name__}. Parameter value: {value!r}."
    )
    result["error_type"] = "type_error:simple"
    return result


def standardize_string(input_string: str):
    """Remove spaces and ",./-_*^", lowercase, and map ' to "."""
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def string_checker(param, model_output, possible_answer):
    standardize_possible_answer = []
    standardize_model_output = standardize_string(model_output)
    for i in range(len(possible_answer)):
        if type(possible_answer[i]) == str:
            standardize_possible_answer.append(standardize_string(possible_answer[i]))
    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [f"Invalid value for parameter {param!r}: {model_output!r}. "
                      f"Expected one of {possible_answer}. Case insensitive."],
            "error_type": "value_error:string",
        }
    return {"valid": True, "error": []}


def list_checker(param, model_output, possible_answer):
    standardize_model_output = list(model_output)
    for i in range(len(standardize_model_output)):
        if type(standardize_model_output[i]) == str:
            standardize_model_output[i] = standardize_string(model_output[i])

    standardize_possible_answer = []
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            if type(possible_answer[i][j]) == str:
                standardize_possible_answer[i].append(standardize_string(possible_answer[i][j]))
            else:
                standardize_possible_answer[i].append(possible_answer[i][j])

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [f"Invalid value for parameter {param!r}: {model_output!r}. "
                      f"Expected one of {possible_answer}."],
            "error_type": "value_error:list/tuple",
        }
    return {"valid": True, "error": []}


def dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):
        if possible_answers[i] == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        possible_answer = possible_answers[i]

        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            standardize_value = value
            if type(value) == str:
                standardize_value = standardize_string(value)

            standardize_possible_answer = []
            for j in range(len(possible_answer[key])):
                if type(possible_answer[key][j]) == str:
                    standardize_possible_answer.append(standardize_string(possible_answer[key][j]))
                else:
                    standardize_possible_answer.append(possible_answer[key][j])

            if standardize_value not in standardize_possible_answer:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {key!r}: {value!r}. "
                    f"Expected one of {standardize_possible_answer}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break

        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


def list_dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for answer_index in range(len(possible_answers)):
        flag = True
        if len(model_output) != len(possible_answers[answer_index]):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue

        for dict_index in range(len(model_output)):
            result = dict_checker(param, model_output[dict_index],
                                  [possible_answers[answer_index][dict_index]])
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def check_param(param_details: dict, param: str, value, acceptable: list):
    """Type and value check for one parameter (the loop body of simple_function_checker).

    `param` must already be known to be in both the schema and the possible answer.
    """
    full_param_details = param_details[param]
    expected_type_description = full_param_details["type"]
    nested_type_converted = None

    expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
    if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
        nested_type = param_details[param]["items"]["type"]
        nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

    if expected_type_description == "tuple" and type(value) == tuple:
        value = list(value)

    # Allow python auto conversion from int to float
    if expected_type_description == "float" and type(value) == int:
        value = float(value)

    type_check_result = type_checker(param, value, acceptable, expected_type_description,
                                     expected_type_converted, nested_type_converted)
    is_variable = type_check_result["is_variable"]
    if not type_check_result["valid"]:
        return type_check_result

    if not is_variable:
        if expected_type_converted == dict:
            return dict_checker(param, value, acceptable)
        elif expected_type_converted == list and nested_type_converted == dict:
            return list_dict_checker(param, value, acceptable)
        elif expected_type_converted == str:
            return string_checker(param, value, acceptable)
        elif expected_type_converted == list:
            return list_checker(param, value, acceptable)

    if value not in acceptable:
        return {
            "valid": False,
            "error": [f"Invalid value for parameter {param!r}: {value!r}. Expected one of {acceptable}."],
            "error_type": "value_error:others",
        }
    return {"valid": True, "error": []}


def simple_function_checker(func_description: dict, model_output: dict, possible_answer: dict):
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"].get("required", [])

    result = {"valid": True, "error": [], "error_type": "simple_function_checker:unclear"}

    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(f"Function name {func_name!r} not found in model output.")
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]

    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {param!r}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {param!r}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result
        r = check_param(param_details, param, value, possible_answer[param])
        if not r["valid"]:
            return r

    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(f"Optional parameter {param!r} not provided and not marked as optional.")
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def parallel_function_checker_no_order(func_descriptions, model_output, possible_answers):
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["Wrong number of functions."],
                "error_type": "parallel_function_checker_no_order:wrong_count"}

    matched_indices = []
    for i in range(len(possible_answers)):
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = find_description(func_descriptions, func_name_expected)
        all_errors = []
        result = {"valid": False}
        for index in range(len(model_output)):
            if index in matched_indices:
                continue
            result = simple_function_checker(func_description, model_output[index], possible_answers[i])
            if result["valid"]:
                matched_indices.append(index)
                break
            all_errors.append({f"Model Result Index {index}": {
                "sub_error": result["error"], "sub_error_type": result["error_type"],
                "model_output_item": model_output[index], "possible_answer_item": possible_answers[i],
            }})

        if not result["valid"]:
            considered_indices = [j for j in range(len(model_output)) if j not in matched_indices]
            all_errors.insert(0, f"Could not find a matching function among index {considered_indices} "
                                 f"of model output for index {i} of possible answers.")
            return {"valid": False, "error": all_errors,
                    "error_type": "parallel_function_checker_no_order:cannot_find_match"}

    return {"valid": True, "error": []}


def multiple_function_checker(func_descriptions, model_output, possible_answers):
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["Wrong number of functions."],
                "error_type": "multiple_function_checker:wrong_count"}
    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = find_description(func_descriptions, func_name_expected)
    return simple_function_checker(func_description, model_output[0], possible_answers[0])


def ast_checker(func_description, model_output, possible_answer, test_category: str):
    """Entry point with the same category dispatch as BFCL.

    `model_output` is a list of `{function_name: {param: value}}` dicts.
    Probe categories are named `probe_parallel` / `probe_parallel_multiple` so the
    same dispatch applies.
    """
    if "parallel" in test_category:
        return parallel_function_checker_no_order(func_description, model_output, possible_answer)
    elif "multiple" in test_category:
        return multiple_function_checker(func_description, model_output, possible_answer)
    else:
        if len(model_output) != 1:
            return {"valid": False, "error": ["Wrong number of functions."],
                    "error_type": "simple_function_checker:wrong_count"}
        return simple_function_checker(func_description[0], model_output[0], possible_answer[0])
