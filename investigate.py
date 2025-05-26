# Importing the OpenROAD library - which is only available when running openroad -python
from openroad import Design, Tech
from odb import *
from pathlib import Path
import utils

print("starting python script...")

# If we had a separate .lef, we could run tech.readLef but we're using a db here that comes packaged with the library.
tech = Tech()
# We do still need to load the Liberty because OpenDB doesn't store enough information for OpenSTA to run correctly.
tech.readLiberty("./corner_libs/NangateOpenCellLibrary_typical.lib")

design = Design(tech) # Every Design has to be associated with a Tech, even if it's empty.

design.readDb("./odbs/3_2_place_iop.odb")
# assert(utils.lib_unit_consistency(design))
library = design.getDb().getLibs()[0]
dbu_per_micron = library.getDbUnitsPerMicron()
cam_vertical_offset = library.getSites()[0].getHeight()
block = design.getBlock()

print(dbu_per_micron)

# export currently placed hypergraph into files

# grab newely placed hypergraph and do rest of placement (but no resizing for some reason?)

# grab metrics of hpwl

# example of how to hightlight a certain cell: select -type Inst -name _486_ -filter Master=INV_X1


# find min, max
# for macro and cells:
    # average h,w, number of pins
    # total amount of 
# num of io pins
# density threshold

# print("block dir:", dir(block))

# print(block.getInsts()[0].getBBox().getLength())
print(block.getInsts()[0].getBBox().yMax() - block.getInsts()[0].getBBox().yMin())

# print("block dir:", dir(block))

# print(dir(block.getInsts()[0]))

# print(block.getInsts()[0].getMaster().isBlock())
m_h = m_w = m_num_pins = m_num = 0
c_h = c_w = c_num_pins = c_num = 0
num_io_pin = len(block.getBTerms())

for inst in block.getInsts():
    if inst.getMaster().isBlock():
        m_num += 1
        m_h += abs(inst.getBBox().yMax() - inst.getBBox().yMin())
        m_w += abs(inst.getBBox().xMax() - inst.getBBox().xMin())
        # m_num_pins += len(inst.getPins())        
    else:
        c_num += 1
        c_h += abs(inst.getBBox().yMax() - inst.getBBox().yMin())
        c_w += abs(inst.getBBox().xMax() - inst.getBBox().xMin())
        # c_num_pins += len(inst.getPins())

if m_num != 0:
    m_num_pins = m_num_pins / m_num
    m_h = m_h / m_num
    m_w = m_w / m_num

if c_num != 0:
    c_num_pins = c_num_pins / c_num
    c_h = c_h / c_num
    c_w = c_w / c_num

print("macros h, w, num_pins, num:\n", m_h, m_w, m_num_pins, m_num)
print("cells h, w, num_pins, num:\n", c_h, c_w, c_num_pins, c_num)
print("num io pins:", num_io_pin)

y = block.getDieArea().yMax() - block.getDieArea().yMin()
x = block.getDieArea().xMax() - block.getDieArea().xMin()
print("die height: ", y)
print("die width: ", x)



# print(len(block.getNets()))
# nets = block.getNets()
# print(dir(nets))
# print(len)
# print(nets[0].getITerms())
# print(nets[0].getBTerms())
# for i in range(len(nets)):/
    # print(nets[i].getName())
    # print(nets[i].getITerms())
    # print(nets[i].getBTerms())


print("finished python script...")