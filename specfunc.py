""" python interface to three kinds of special functions:

    hyp2f1 - the Gauss hypergeometric function (for 0<|z|< 1!)
    hyp1f1 - confluent hypergeometric function (for 0<|z|< 1!)
    pcfd - parabolic cylinder functions

the bonus in comparison to most existing implementations is:
    - they can operate on numpy arrays
    - they take complex arguments everywhere

if libspecfunc.so is found, fast c implementations are used 
(see specfunc.c), otherwise the system falls back to the slower
but excellent mpmath python library

libspecfunc.so needs to be in the same directory as this module!
"""

import numpy as np
import numpy.ctypeslib as npct
from ctypes import *
import os
import mpmath as mp

array_1d_complex = npct.ndpointer(dtype=np.complex128, ndim=1, flags='CONTIGUOUS')

class Complex(Structure):
    _fields_ = [("re", c_double), ("im", c_double)]

class PrmsAndInfo(Structure):
    _fields_ = [("max_iter", c_int), ("tol", c_double),
            ("iters_needed", c_int), ("tol_achieved", c_double),
            ("prec_warning", c_int)]

def cmpl(val):
    return Complex(c_double(val.real), c_double(val.imag))

max_iter = 5000
tol = 1e-15
use_mpmath = False
nofallback = False
#mp = None

lib = None
libname = "libspecfunc.so"
try:
    lib = npct.load_library(libname, os.path.dirname(os.path.abspath(__file__)))
except OSError as e:
    print e
    print("cannot load %s, falling back to mpmath..." % libname)
    use_mpmath = True
    #mp = __import__("mpmath")

if lib is not None:
    # hyp1f1
    lib.hyp1f1.restype = Complex
    lib.hyp2f1.argtypes = [Complex, Complex, Complex, POINTER(PrmsAndInfo)]
    # hyp1f1_a_arr
    lib.hyp1f1_a_arr.restype = None
    lib.hyp1f1_a_arr.argtypes = [array_1d_complex, Complex, Complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]
    # hyp1f1_b_arr
    lib.hyp1f1_b_arr.restype = None
    lib.hyp1f1_b_arr.argtypes = [Complex, array_1d_complex, Complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]
    # hyp1f1_z_arr
    lib.hyp1f1_z_arr.restype = None
    lib.hyp1f1_z_arr.argtypes = [Complex, Complex, array_1d_complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]
    # hyp1f1_all_arr
    lib.hyp1f1_all_arr.restype = None
    lib.hyp1f1_all_arr.argtypes = [array_1d_complex, array_1d_complex, array_1d_complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]

    # hyp2f1
    lib.hyp2f1.restype = Complex 
    lib.hyp2f1.argtypes = [Complex, Complex, Complex, Complex, POINTER(PrmsAndInfo)]
    # hyp2f1_a_arr
    lib.hyp2f1_a_arr.restype = None
    lib.hyp2f1_a_arr.argtypes = [array_1d_complex, Complex, Complex, Complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]
    # hyp2f1_b_arr
    lib.hyp2f1_b_arr.restype = None
    lib.hyp2f1_b_arr.argtypes = [Complex, array_1d_complex, Complex, Complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]
    # hyp2f1_c_arr
    lib.hyp2f1_c_arr.restype = None
    lib.hyp2f1_c_arr.argtypes = [Complex, Complex, array_1d_complex, Complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]
    # hyp2f1_z_arr
    lib.hyp2f1_z_arr.restype = None
    lib.hyp2f1_z_arr.argtypes = [Complex, Complex, Complex, array_1d_complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]
    # hyp2f1_all_arr
    lib.hyp2f1_all_arr.restype = None
    lib.hyp2f1_all_arr.argtypes = [array_1d_complex, array_1d_complex, array_1d_complex, array_1d_complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]

    # pcdf
    lib.pcfd.restype = Complex
    lib.pcfd.argtypes = [Complex, Complex]
    # pcdf_nu_arr
    lib.pcfd_nu_arr.restype = None
    lib.pcfd_nu_arr.argtypes = [array_1d_complex, Complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]
    # pcdf_z_arr
    lib.pcfd_z_arr.restype = None
    lib.pcfd_z_arr.argtypes = [Complex, array_1d_complex, array_1d_complex, c_int, POINTER(PrmsAndInfo)]

def hyp1f1(a, b, z):
    #global mp
    """ Computes the confluent hypergeometric function.

    The parameters a, b, and z may be complex. Further, one or more of them may be numpy arrays. 
    """
    uselib = lib is not None and not use_mpmath
    #if not uselib and mp is None:
    #    mp = __import__("mpmath")

    p = PrmsAndInfo(c_int(max_iter), c_double(tol), c_int(0), c_double(0), c_int(0))
    if (np.ndim(a) + np.ndim(b) + np.ndim(z) > 1):
        l = [len(x) for x in (a, b, z) if hasattr(x, "__len__")]
        if l[1:] != l[:-1]:
            raise TypeError("if more than one parameter is a numpy array, they have to have the same length")
        a, b, z = [np.ones(l[0])*x if not hasattr(x, "__len__") else x for x in (a, b, z)]
        if uselib:
            out = np.zeros(l[0], dtype=np.complex128)
            lib.hyp1f1_all_arr(a.astype(np.complex128), b.astype(np.complex128), z.astype(np.complex128), out, len(out), byref(p))
        if not nofallback and p.prec_warning or not uselib:
            out = np.array([mp.hyp1f1(aa, bb,  zz) for aa, bb, zz in zip(a, b, z)], dtype=np.complex128)
        return out
    if (np.ndim(a) == 1):
        if uselib:
            out = np.zeros(len(a), dtype=np.complex128)
            lib.hyp1f1_a_arr(a.astype(np.complex128), cmpl(b), cmpl(z), out, len(out), byref(p))
        if not nofallback and p.prec_warning or not uselib:
            out = np.array([mp.hyp1f1(aa, b, z) for aa in a], dtype=np.complex128)
        return out
    elif (np.ndim(b) == 1):
        if uselib:
            out = np.zeros(len(b), dtype=np.complex128)
            lib.hyp1f1_b_arr(cmpl(a), b.astype(np.complex128), cmpl(z), out, len(out), byref(p))
        if not nofallback and p.prec_warning or not uselib:
            out =  np.array([mp.hyp1f1(a, bb, z) for bb in b], dtype=np.complex128)
        return out
    elif (np.ndim(z) == 1):
        if uselib:
            out = np.zeros(len(z), dtype=np.complex128)
            lib.hyp1f1_z_arr(cmpl(a), cmpl(b), z.astype(np.complex128), out, len(out), byref(p))
        if not nofallback and p.prec_warning or not uselib:
            out = np.array([mp.hyp1f1(a, b, zz) for zz in z], dtype=np.complex128)
        return out
    else: 
        if uselib:
            c = lib.hyp1f1(cmpl(a), cmpl(b), cmpl(z), byref(p))
            out = c.re + 1j* c.im
        if not nofallback and p.prec_warning or not uselib:
            out = np.complex128(mp.hyp1f1(a, b, z))
        return out

# XXX TODO: check for bad prec_value and fall back to mpmath
def hyp2f1(a, b, c, z):
    """ Computes the Gauss hypergeometric function.

    The parameters a, b, c, and z may be complex. Further, one or more of them may be numpy arrays.
    """
    uselib = lib is not None and not use_mpmath
    #if not uselib and mp is None:
    #    mp = __import__("mpmath")
    
    p = PrmsAndInfo(c_int(max_iter), c_double(tol), c_int(0), c_double(0))
    if (np.ndim(a) + np.ndim(b) + np.ndim(c) + np.ndim(z) > 1):
        l = [len(x) for x in (a, b, c, z) if hasattr(x, "__len__")]
        if l[1:] != l[:-1]:
            raise TypeError("if more than one parameter is a numpy array, they have to have the same length")
        a, b, c, z = [np.ones(l[0])*x if not hasattr(x, "__len__") else x for x in (a, b, c, z)]
        if uselib:
            out = np.zeros(l[0], dtype=np.complex128)
            lib.hyp2f1_all_arr(a.astype(np.complex128), b.astype(np.complex128), c.astype(np.complex128), z.astype(np.complex128), out, len(out), byref(p))
            return out
        else:
            return np.array([mp.hyp2f1(aa, bb, cc, zz) for aa, bb, cc, zz in zip(a, b, c, z)], dtype=np.complex128)
    if (np.ndim(a) == 1):
        if uselib:
            out = np.zeros(len(a), dtype=np.complex128)
            lib.hyp2f1_a_arr(a.astype(np.complex128), cmpl(b), cmpl(c), cmpl(z), out, len(out), byref(p))
            return out
        else:
            return np.array([mp.hyp2f1(aa, b, c, z) for aa in a], dtype=np.complex128)
    elif (np.ndim(b) == 1):
        if uselib:
            out = np.zeros(len(b), dtype=np.complex128)
            lib.hyp2f1_b_arr(cmpl(a), b.astype(np.complex128), cmpl(c), cmpl(z), out, len(out), byref(p))
            return out
        else:
            return np.array([mp.hyp2f1(a, bb, c, z) for bb in b], dtype=np.complex128)
    elif (np.ndim(c) == 1):
        if uselib:
            out = np.zeros(len(c), dtype=np.complex128)
            lib.hyp2f1_c_arr(cmpl(a), cmpl(b), c.astype(np.complex128), cmpl(z), out, len(out), byref(p))
            return out
        else:
            return np.array([mp.hyp2f1(a, b, cc, z) for cc in c], dtype=np.complex128)
    elif (np.ndim(z) == 1):
        if uselib:
            out = np.zeros(len(z), dtype=np.complex128)
            lib.hyp2f1_z_arr(cmpl(a), cmpl(b), cmpl(c), z.astype(np.complex128), out, len(out), byref(p))
            return out
        else:
            return np.array([mp.hyp2f1(a, b, c, zz) for zz in z], dtype=np.complex128)
    else: 
        if uselib:
            res = lib.hyp2f1(cmpl(a), cmpl(b), cmpl(c), cmpl(z), byref(p))
            #print "p.tol_achieved", p.tol_achieved, "p.iters_needed", p.iters_needed
            return res.re + 1j* res.im
        else:
            return np.complex128(mp.hyp2f1(a, b, c, z))

def pcfd(nu, z, ):
    """ Computes the parabolic cylinder function.

    The parameters nu and z may be complex. Further, one of them may be a numpy array.
    This always uses mpmath for the time being as the c implementation is unreliable.

    """
    uselib = lib is not None and not use_mpmath
    #if not uselib and mp is None:
    #    mp = __import__("mpmath")

    if (np.ndim(nu) + np.ndim(z) > 1):
        raise TypeError("at most one parameter may be a numpy array")
    if False: # if lib is not None and not use_mpmath:
        p = PrmsAndInfo(c_int(max_iter), c_double(tol), c_int(0), c_double(0))
        if (np.ndim(nu) == 1):
            out = np.zeros(len(nu), dtype=np.complex128)
            lib.pcfd_nu_arr(nu.astype(np.complex128), cmpl(z), out, len(out), byref(p))
            return out
        elif (np.ndim(z) == 1):
            out = np.zeros(len(z), dtype=np.complex128)
            lib.pcfd_z_arr(cmpl(nu), z.astype(np.complex128), out, len(out), byref(p))
            return out
        else:
            c = lib.pcfd2(cmpl(nu), cmpl(z), byref(p))
            return c.re + 1j* c.im
    else:
        if (np.ndim(nu) == 1):
            return np.array([np.complex128(mp.pcfd(nunu, z)) for nunu in nu])
        elif (np.ndim(z) == 1):
            return np.array([np.complex128(mp.pcfd(nu, zz)) for zz in z])
        else:
            return np.complex128(mp.pcfd(nu, z))
            
    


