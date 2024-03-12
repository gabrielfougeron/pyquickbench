
import os
import sys

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)


import pyquickbench
import time

TT = pyquickbench.TimeTrain(
    names_reduction = "random_el",
)


time.sleep(0.01)
TT.toc("a")

time.sleep(0.02)
TT.toc("a")

time.sleep(0.03)
TT.toc("a")

time.sleep(0.04)
TT.toc("a")

time.sleep(0.1)
TT.toc("b")



print(TT)

d = TT.to_dict()
print(d)


print(len(TT))