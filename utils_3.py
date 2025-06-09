from typing import List, Tuple

from dataclasses import dataclass, field


@dataclass
class PinSlot:
    dx: float
    dy: float

    def __str__(self):
        return f"(dx, dy): ({self.dx}, {self.dy})"
    
@dataclass
class Macro:
    width: float
    height: float
    pin_slots: List[PinSlot]
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

@dataclass
class StandardCell:
    width: float
    height: float
    pin_slots: List[PinSlot]
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

@dataclass
class IOPin:
    width: float
    height: float
    x: float = field(init=False, default=None)
    y: float = field(init=False, default=None)

class ChipArea:
    def __init__(self, width, height, macros, std_cells, io_pins, rng=None):
        self.width, self.height = width, height
        self.macros = macros
        self.std_cells = std_cells
        self.io_pins = io_pins
        self.rng = rng
        # non_normalized variables (designated with "nn")
        self.nn_width = None
        self.nn_height = None

def normalize_object(object, max_1_half):
    object.width = object.width / max_1_half
    object.height = object.height / max_1_half
    for pin in object.pin_slots:
        pin.dx = pin.dx / max_1_half
        pin.dy = pin.dy / max_1_half

def normalize_dimensions(chip_h, chip_w, macros: List[Macro], cells: List[StandardCell], ios: List[IOPin]):
    max_1_half = max(chip_h, chip_w) / 2
    # divide all dimensions by max_1_half
    norm_chip_h = chip_h / max_1_half
    norm_chip_w = chip_w / max_1_half
    for macro in macros:
        normalize_object(macro, max_1_half)
    for cell in cells:
        normalize_object(cell, max_1_half)
    for pin in ios:
        pin.width = pin.width / max_1_half
        pin.height = pin.height / max_1_half
    return (norm_chip_h, norm_chip_w, macros, cells, ios)

def shift_norm_object(object, norm_shift_left_amount, norm_shift_down_amount):
    object.x = object.x - norm_shift_left_amount
    object.y = object.y - norm_shift_down_amount

def center_normalized_placement(chip: ChipArea):
    if (chip.height > chip.width):
        norm_shift_left_amount = chip.width / 2
        norm_shift_down_amount = 1
    else:
        norm_shift_left_amount = 1
        norm_shift_down_amount = chip.height / 2
    # center chip
    for macro in chip.macros:
        shift_norm_object(macro, norm_shift_left_amount, norm_shift_down_amount)
    for cell in chip.std_cells:
        shift_norm_object(cell, norm_shift_left_amount, norm_shift_down_amount)
    for pin in chip.io_pins:
        shift_norm_object(pin, norm_shift_left_amount, norm_shift_down_amount)
    return chip