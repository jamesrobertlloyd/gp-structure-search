
import sys
import ctypes

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)
import scipy.linalg
sys.setdlopenflags(flags)
