import os
import sys


import numpy as np
import mpmath


# ###############################
# ###############################
#       DEFAULT
# ###############################


# DEFAULT PATH and PFE (= Path/File.Ext)
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
    details=None,
):
    """ Compute hyp2f1() on a large set of test data

    - with specfunc.hyp2f1()
    - with mpmath.hyp2f1()

    And checks for the presence of inf, nan or differences in the results
    Raise an Exception detailing the number of each case, if any

    Arg subset can be used to limit the analysis to a user-provided index

    """

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

    # details
    if details is None:
        details = not (subset is None)
    assert isinstance(details, bool)

    # -------------------
    # load local MWE file
    # -------------------

    dout = {k0: v0 for k0, v0 in np.load(pfe_mwe).items()}

    # optional subset for test
    ind0 = np.arange(dout['a'].size)
    if subset is not None:
        for k0, v0 in dout.items():
            dout[k0] = v0[subset]
        ind0 = ind0[subset]

    # -------------------
    # call specfunc.hyp2F1
    # -------------------

    if verb is True:
        msg = "\nComputing specfunc.hyp2f1()"
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
            print(msg, end="\n\n" if ii == size-1 else "\r")

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
    ifail = (iinf | inan)
    ifail[iok] = idiff

    # exception
    if any(iinf | inan) or np.any(idiff):

        # details
        if details is True:
            vspec = out_specfunc[ifail].astype(str)
            vmath = out_mpmath[ifail].astype(str)
            spec_max = np.max(np.char.str_len(vspec))
            math_max = np.max(np.char.str_len(vmath))
            vspec = np.char.rjust(vspec, spec_max)
            vmath = np.char.rjust(vmath, math_max)

            lstr = [
                f"\t- {ind0[ifail[ii]]}: {vspec[ii]} vs {vmath[ii]}"
                for ii in range(ifail.sum())
            ]
            # lmax = np.max([len(ss) for ss in lstr])
            msg_details = (
                " "*len(f"\t\t- {ind0[-1]}")
                + "specfunc\tvs\tout_mpmath\n"
                + "\n".join(lstr)
            )
        else:
            msg_details = ''

        # msg
        msg = (
            f"out_specfunc, of size = {size}, contains:\n"
            f"\t- inf: {iinf.sum()}\n"
            f"\t- nan: {inan.sum()}\n"
            f"\t- diff from mp.math: {idiff.sum()}\n"
            + msg_details
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
