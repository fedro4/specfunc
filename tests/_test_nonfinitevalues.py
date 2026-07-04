import os
import sys


import numpy as np
import mpmath


# ###############################
# ###############################
#       DEFAULT
# ###############################


_PATH_HERE = os.path.dirname(__file__)
_PFE_MWE = os.path.join(_PATH_HERE, 'specfunc_MWE.npz')
_PATH_PROJECT = os.path.dirname(_PATH_HERE)


# Make sure to load the local version of specfunc
# => temporarilly insert local pah to sys.path in  1st position
sys.path.insert(0, _PATH_PROJECT)
import specfunc
sys.path.pop(0)


# ###############################
# ###############################
#       main
# ###############################


def main(
    # subset of test data
    subset=None,
    # path/file.ext to MWE
    pfe_mwe=None,
    verb=None,
):

    # -------------------
    # check inputs
    # -------------------

    # subset => must be a valid index
    if subset is not None:
        try:
            _ = np.ones((70000,), dtype=float)[subset]
            assert isinstance(_, np.ndarray)
        except Exception as err:
            msg = "Arg subset must be a valid index to a 1d array"
            raise Exception(msg)

    # pfe_mwe => path/file.ext to existing mwe data file
    if pfe_mwe is None:
        pfe_mwe = _PFE_MWE
    assert os.path.isfile(pfe_mwe) and pfe_mwe.endswith('.npz')

    # optional verbosity
    if verb is None:
        verb = True
    assert isinstance(verb, bool)

    # -------------------
    # load local MWE file
    # -------------------

    dout = {k0: v0 for k0, v0 in np.load(pfe_mwe).items()}

    # optional subset for test
    if subset is not None:
        for k0, v0 in dout.items():
            dout[k0] = v0[subset]

    # -------------------
    # call specfunc.hyp2F1
    # -------------------

    if verb is True:
        msg = "Computing specfunc.hyp2f1()"
        print(msg)

    out_specfunc = specfunc.hyp2f1(
        dout['a'],
        dout['b'],
        dout['c'],
        dout['z'],
    )

    # -------------------
    # call npmath
    # -------------------

    shape = dout['a'].shape
    size = dout['a'].size
    just = len(f"mpmath.hyp2f1() on value {size} / {size}")
    out_mpmath = np.zeros(shape, dtype=out_specfunc.dtype)
    for ii, ind in enumerate(np.ndindex(shape)):

        if verb is True:
            msg = f"mpmath.hyp2f1() on value {ii+1} / {size}".ljust(just)
            print(msg, end="\n" if ii == size-1 else "\r")

        out_mpmath[ind] = mpmath.hyp2f1(
            dout['a'][ind],
            dout['b'][ind],
            dout['c'][ind],
            dout['z'][ind],
        )

    # -------------------
    # compare values
    # -------------------

    # indexes of fails
    iinf = np.isinf(out_specfunc)
    inan = np.isnan(out_specfunc)
    iok = ~(iinf | inan)
    idiff = (out_specfunc[iok] != out_mpmath[iok])

    # exception
    if any(iinf | inan) or np.any(idiff):
        msg = (
            f"out_specfunc, of size = {size}, contains:\n"
            f"\t- inf: {iinf.sum()}\n"
            f"\t- nan: {inan.sum()}\n"
            f"\t- diff from mp.math: {idiff.sum()}\n"
        )
        raise Exception(msg)

    return


# ###############################
# ###############################
#       __main__
# ###############################


# in case we want to run from terminal
if __name__ == '__main__':
    main()
