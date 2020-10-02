import ctypes,os.path
ctypes.CDLL(os.path.join(os.path.dirname(__file__),"libcilkrts.so.5"))
ctypes.CDLL(os.path.join(os.path.dirname(__file__),"libmosek64.so.9.2"))
