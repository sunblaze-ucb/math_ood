import pdb
from geo_types import AngleSize
from typing import Tuple, List, Optional
import math
from dataclasses import dataclass

def format_vertex_list(vertex_list: List[str]) -> str:
    if len(vertex_list) == 0:
        return ""
    elif len(vertex_list) == 1:
        return vertex_list[0]
    else:
        new_vertex_list = vertex_list.copy()
        new_vertex_list[-1] = "and " + new_vertex_list[-1]
        return ", ".join(new_vertex_list)

def invert_pi_expression(value):
    if isinstance(value, AngleSize):
        value = value.x
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            # for idempotency
            return value
    if abs(value - math.pi) < 1e-4:
        return "pi"
    elif abs(value - math.pi / 2) < 1e-4:
        return "pi/2"
    elif abs(value - math.pi / 3) < 1e-4:
        return "pi/3"
    elif abs(value - 2 * math.pi / 3) < 1e-4:
        return "2pi/3"
    elif abs(value - math.pi / 4) < 1e-4:
        return "pi/4"
    elif abs(value - 3 * math.pi / 4) < 1e-4:
        return "3pi/4"
    elif abs(value - math.pi / 5) < 1e-4:
        return "pi/5"
    elif abs(value - 2 * math.pi / 5) < 1e-4:
        return "2pi/5"
    elif abs(value - 3 * math.pi / 5) < 1e-4:
        return "3pi/5"
    elif abs(value - 4 * math.pi / 5) < 1e-4:
        return "4pi/5"
    elif abs(value - math.pi / 6) < 1e-4:
        return "pi/6"
    elif abs(value - 5 * math.pi / 6) < 1e-4:
        return "5pi/6"
    elif abs(value - math.pi / 7) < 1e-4:
        return "pi/7"
    elif abs(value - 2 * math.pi / 7) < 1e-4:
        return "2pi/7"
    elif abs(value - 3 * math.pi / 7) < 1e-4:
        return "3pi/7"
    elif abs(value - 4 * math.pi / 7) < 1e-4:
        return "4pi/7"
    elif abs(value - 5 * math.pi / 7) < 1e-4:
        return "5pi/7"
    elif abs(value - 6 * math.pi / 7) < 1e-4:
        return "6pi/7"
    elif abs(value - math.pi / 8) < 1e-4:
        return "pi/8"
    elif abs(value - 3 * math.pi / 8) < 1e-4:
        return "3pi/8"
    elif abs(value - 5 * math.pi / 8) < 1e-4:
        return "5pi/8"
    elif abs(value - 7 * math.pi / 8) < 1e-4:
        return "7pi/8"
    elif abs(value - math.pi / 9) < 1e-4:
        return "pi/9"
    elif abs(value - 2 * math.pi / 9) < 1e-4:
        return "2pi/9"
    elif abs(value - 4 * math.pi / 9) < 1e-4:
        return "4pi/9"
    elif abs(value - 5 * math.pi / 9) < 1e-4:
        return "5pi/9"
    elif abs(value - 7 * math.pi / 9) < 1e-4:
        return "7pi/9"
    elif abs(value - 8 * math.pi / 9) < 1e-4:
        return "8pi/9"
    elif abs(value - math.pi / 10) < 1e-4:
        return "pi/10"
    elif abs(value - 3 * math.pi / 10) < 1e-4:
        return "3pi/10"
    elif abs(value - 7 * math.pi / 10) < 1e-4:
        return "7pi/10"
    elif abs(value - 9 * math.pi / 10) < 1e-4:
        return "9pi/10"
    elif abs(value - math.pi / 11) < 1e-4:
        return "pi/11"
    elif abs(value - 2 * math.pi / 11) < 1e-4:
        return "2pi/11"
    elif abs(value - 3 * math.pi / 11) < 1e-4:
        return "3pi/11"
    elif abs(value - 4 * math.pi / 11) < 1e-4:
        return "4pi/11"
    elif abs(value - 5 * math.pi / 11) < 1e-4:
        return "5pi/11"
    elif abs(value - 6 * math.pi / 11) < 1e-4:
        return "6pi/11"
    elif abs(value - 7 * math.pi / 11) < 1e-4:
        return "7pi/11"
    elif abs(value - 8 * math.pi / 11) < 1e-4:
        return "8pi/11"
    elif abs(value - 9 * math.pi / 11) < 1e-4:
        return "9pi/11"
    elif abs(value - 10 * math.pi / 11) < 1e-4:
        return "10pi/11"
    elif abs(value - math.pi / 12) < 1e-4:
        return "pi/12"
    elif abs(value - 5 * math.pi / 12) < 1e-4:
        return "5pi/12"
    elif abs(value - 7 * math.pi / 12) < 1e-4:
        return "7pi/12"
    elif abs(value - 11 * math.pi / 12) < 1e-4:
        return "11pi/12"
    else:
        return "%0.6f" % value

@dataclass
class Command:
    name: str
    inputs: Tuple[str, ...]
    output: Optional[str] = None
