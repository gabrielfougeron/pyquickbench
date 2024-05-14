import pyquickbench
import time

TT = pyquickbench.TimeTrain(name="Toto", include_locs=False, names_reduction="sum")

@TT.tictoc
def wait(n):
    time.sleep(n)    

def cantwait(n):
    time.sleep(n)    
    
wait(0.1)
wait(0.1)

cantwait(0.3)
TT.toc("cantwait")

wait(0.2)

print(TT)