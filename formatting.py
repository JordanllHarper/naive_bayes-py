from typing import Callable


def space():
    print()


def sep():
    print("--------")


def print_with_header(label):
    print("Task:", label)


def step_print(
    step_num: int,
    message: str
):
    sep()
    print("[STEP {step_num}]: {msg}\n".format(step_num=step_num, msg=message))


def process_and_print(
        label: str,
        process: Callable
):
    print_with_header(label)
    result = process()
    print(result)
    space()
    return result
