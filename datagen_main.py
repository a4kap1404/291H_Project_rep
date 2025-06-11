"""
    Generates N placements
        - Based on cell size range, avg macro to cell size ratio, avg chip size range, and other hyperparams (can modify these)
            generates a placement 

    Note: Will take a long time, scales O(n^2) with num of cells, but can adjust certain hyperparams to speed it up
"""


import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import List, Tuple
import pickle
import numpy as np
import os

from py_utils.datagen import *


if __name__ == '__main__':

    debug = False
    save = True
    # do_plot = True # dont keep this True
    do_plot = False
    # small = True # dont keep this True
    small = False

    N = 30 # num of placements
    filename = "syn_data/Data" # will add "_N{N}_v{version number}" if over N=5 samples, to not accidentally overwrite data
    seed = 40
    density = 0.9
    max_macro_retries = 30 # can reduce retries to speed up, but this already pretty low
    max_cell_retries = 30
    div = 10000 # increasing will reduce size of square region used to calculate density
    # scale = 4.0
    scale = 0.4 # increasing will increase likelyhood that a connection that will be made will be 2 pins that are farther apart
    

    # edge sample ratio: smaller means faster, but produces less edges
    edge_sample_ratio = 20

    # bad name? # make bigger to run faster, but less edges
    edge_to_node_ratio = 10


    """ changing these too much could produce too large of cells, making 
        it hard to place, increasing runtime to produce data
    """
    macro_h_and_w_dev_factor = 0.2
    macro_pin_dev_factor = 0.2
    cell_h_and_w_dev_factor = 0.2
    cell_pin_dev_factor = 0.2

    cluster_h_and_w_dev_factor = None
    cluster_pin_dev_factor = None

    # From cell metrics
    MIN_C_H = 2800
    MAX_C_H = 2800

    MIN_C_W = 1718.62   # from GCD
    MAX_C_W = 1991.75   # from AES
    # MAX_C_W = 1718.62   # from GCD

    MIN_C_NUM_PINS = 2
    # MAX_C_NUM_PINS = 0  # since all are 0
    MAX_C_NUM_PINS = 10 

    MIN_C_NUM = 551     # GCD
    MAX_C_NUM = 6000   # manual
    # MAX_C_NUM = 15478   # AES
    # MAX_C_NUM = 15478   # AES
    # MAX_C_NUM = 551     # GCD

    # manually set
    cell_to_macro_size = 8
    # MIN_M_H = 0
    MIN_M_H = int(cell_to_macro_size * MIN_C_H)
    # MAX_M_H = 0
    MAX_M_H = int(cell_to_macro_size * MAX_C_H)
    # MIN_M_W = 0
    MIN_M_W = int(cell_to_macro_size * MIN_C_W)
    # MAX_M_W = 0
    MAX_M_W = int(cell_to_macro_size * MAX_C_W)
    MIN_M_NUM_PINS = int(MIN_C_NUM_PINS * 2)
    MAX_M_NUM_PINS = int(MAX_C_NUM_PINS * 2)
    MIN_M_NUM = 0
    MAX_M_NUM = 10


    # Chip dimensions
    if small:
        MIN_CHIP_W = 20000
        MAX_CHIP_W = 20000
        

        MIN_CHIP_H = 20000
        MAX_CHIP_H = 20000
    else:
        MIN_CHIP_W = 72760     # GCD
        MAX_CHIP_W = 100000    # CUSTOM
        # MAX_CHIP_W = 499700    # AES
        # MAX_CHIP_W = 72760     # GCD
        

        MIN_CHIP_H = 72760     # GCD
        MAX_CHIP_H = 100000    # CUSTOM
        # MAX_CHIP_H = 501200    # AES
        # MAX_CHIP_H = 72760     # GCD

    if N > 30 and save == False:
        print("Error, N is", N, ", but save is False")
        exit()


    AVG_IO_PIN = 22

    Data = list()

    for i in range(N):

        print("\t--------------", i, "---------------")

        random.seed(seed + i) # also IDK if this is correct
        # WHAT ABOUT THE NUMPY RANDOMNESS SEED

        # compute average values
        m_h = random.randint(MIN_M_H, MAX_M_H)
        m_w = random.randint(MIN_M_W, MAX_M_W)
        m_num_pins = random.randint(MIN_M_NUM_PINS, MAX_M_NUM_PINS)
        # m_num = random.randint(MIN_M_NUM, MAX_M_NUM)

        c_h = random.randint(MIN_C_H, MAX_C_H)
        c_w = int(random.uniform(MIN_C_W, MAX_C_W))
        c_num_pins = random.randint(MIN_C_NUM_PINS, MAX_C_NUM_PINS)
        # c_num = random.randint(MIN_C_NUM, MAX_C_NUM)

        num_io_pin = AVG_IO_PIN # CHANGE if needed

        chip_w = random.randint(MIN_CHIP_W, MAX_CHIP_W)
        chip_h = random.randint(MIN_CHIP_H, MAX_CHIP_H)

        # div = 100

        # estimate num of macros and cells
        num_cells, num_macros = estimate_num_cells_and_area(density, chip_h, chip_w, c_h, c_w, m_h, m_w)

        print("num_cells", num_cells)
        print("num_macros", num_macros)

        macros = create_macros(m_h, MIN_M_H, MAX_M_H, 
                               m_w, MIN_M_W, MAX_M_W, 
                               m_num_pins, MIN_M_NUM_PINS, MAX_M_NUM_PINS,
                               num_macros, 
                               macro_h_and_w_dev_factor, macro_pin_dev_factor)
        
        cells = create_std_cells(c_h, MIN_C_H, MAX_C_H, 
                                 c_w, MIN_C_W, MAX_C_W, 
                                 c_num_pins, MIN_C_NUM_PINS, MAX_C_NUM_PINS,
                                 num_cells, 
                                 cell_h_and_w_dev_factor, cell_pin_dev_factor)

        ios = [IOPin(1, 1) for _ in range(num_io_pin)]

        # normalization
        norm_chip_h, norm_chip_w, macros, cells, ios = normalize_dimensions(
            chip_h, chip_w, macros, cells, ios)

        # placement creation
        chip, G = generate_placement(
            norm_chip_w, norm_chip_h, macros, cells, ios, norm_chip_h / div, 
            density, steps=50, scale=scale, seed=seed, 
            max_macro_retries=max_macro_retries, max_cell_retries=max_cell_retries, edge_sample_ratio=edge_sample_ratio,
            edge_to_node_ratio=edge_to_node_ratio)

        # set non normalized variables to allow for recreation    
        chip.nn_width = chip_w
        chip.nn_height = chip_h

        if (debug):
            # assert(chip_good(chip))
            print_cells_attr(chip, cell_count=5, macro_count=2, io_count=2)
            chip_good_uncentered(chip, debug)
            
        if (do_plot):
            if i == 0:
                plot_full(chip, G)


        chip = center_normalized_placement(chip)


        for cell in chip.std_cells:
            if cell.x is None or cell.y is None:
                raise Exception("cell has None positions!")
        Data.append((chip, G))

    if save:
        print("--SAVING--")
        if (N > 5):
            i = 0
            base_filename = f"{filename}_N{N}"
            filename = f"{base_filename}_v{i}"
            while os.path.exists(filename + ".pkl"):
                i += 1
            filename = f"{base_filename}_v{i}"

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(Data, f)


    # with open("Data.pkl", "rb") as f:
        # Data = pickle.load(f)