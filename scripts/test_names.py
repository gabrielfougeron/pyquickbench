import inspect

def toto(a,b=1):
    return None

sig = inspect.signature(toto)

print(inspect.signature(toto))
print(type(sig))
print(dir(sig))


sig_str = str(sig)
print(sig_str)