#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

__version__ = "0.0.2"

import os
import sys
import ctypes
import numpy as np

_DBG = False

class Context():
    """
    Blacs context wrapper.
    """

    def __init__(self, scalapack, layout, nprow, npcol):
        """
        Create a blacs context.

        Note: create Context object will not create blacs grid immediately.
        User should use this type via "with" statement like the following.
        ```
        with Context(scalapack, layout, nprow, npcol) as context:
           ...
        ```

        Parameters
        ----------
        scalapack : Scalapack
            The scalapack library handle.
        layout : b'R' | b'C'
            The layout of the blacs grid, row major or column major.
        nprow, npcol : int
            The row and column of the blacs grid.

        See Also
        --------
        Scalapack.__call__ : Create blacs Context from scalapck library handle.
        """
        self.scalapack = scalapack
        self.layout = ctypes.c_char(layout)
        self.nprow = ctypes.c_int(nprow)
        self.npcol = ctypes.c_int(npcol)

    def barrier(self, scope=b'A'):
        """
        Blacs barrier.

        Parameters
        ----------
        scope : b'A' | b'R' | b'C'
            The scope to barrier.
        """
        if scope not in [b'A', b'R', b'C']:
            raise RuntimeError(f"scope should be b'A', b'R' or b'C', but it is {scope}.")
        if self:
            self.scalapack.blacs_barrier(self.ictxt, ctypes.c_char(scope))

    def _call_blacs_pinfo(self):
        """
        Get the rank and size of total task.
        """
        self.rank = ctypes.c_int()
        self.size = ctypes.c_int()
        self.scalapack.blacs_pinfo(self.rank, self.size)

    def _call_blacs_get(self):
        """
        Get the blacs context.
        """
        self.ictxt = ctypes.c_int()
        self.scalapack.blacs_get(Scalapack.neg_one, Scalapack.zero, self.ictxt)

    def _call_blacs_gridinit(self):
        """
        Create blacs grid.
        """
        if self.nprow.value == -1:
            self.nprow = ctypes.c_int(self.size.value // self.npcol.value)
        if self.npcol.value == -1:
            self.npcol = ctypes.c_int(self.size.value // self.nprow.value)
        self.scalapack.blacs_gridinit(self.ictxt, self.layout, self.nprow, self.npcol)

    def _call_blacs_gridinfo(self):
        """
        Get the grid info.
        """
        self.myrow = ctypes.c_int()
        self.mycol = ctypes.c_int()
        self.scalapack.blacs_gridinfo(self.ictxt, self.nprow, self.npcol, self.myrow, self.mycol)

    def __enter__(self):
        """
        Enter the Context and create the blacs grid.
        """
        self._call_blacs_pinfo()
        self._call_blacs_get()
        self._call_blacs_gridinit()
        self._call_blacs_gridinfo()
        # self.valid check if the current process is in the grid created.
        self.valid = self.rank.value < self.nprow.value * self.npcol.value
        return self

    def _call_blacs_gridexit(self):
        """
        Destroy the blacs grid.
        """
        self.scalapack.blacs_gridexit(self.ictxt)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the Context.
        """
        # If the current process is in the grid, destroy the grid.
        if self.valid:
            self._call_blacs_gridexit()
        if exc_type is not None:
            return False

    def __bool__(self):
        """
        Check if the current process is in the grid of the current context.

        Returns
        -------
        bool
            Whether the current process is in the grid.
        """
        return self.valid

    def array(self, m, n, mb, nb, *, data=None, dtype=None):
        """
        Create an blacs array in this Context. User must specify either `data` or `dtype`.

        Parameters
        ----------
        m, n : int
            The row and column of the array.
        mb, nb : int
            The block size on row and column.
        data : np.ndarray, optional
            The created blacs array does not owndata, and just use the data in the given numpy array.
        dtype : np.dtype, optional
            Create an new numpy array as the data of the created blacs array.

        Returns
        -------
        Array
            The result blacs array.
        """
        if data is None and dtype is None:
            raise RuntimeError("Either `data` or `dtype` should be specified")
        return Array(self, m, n, mb, nb, data=data, dtype=dtype)


class ArrayDesc(ctypes.Structure):
    """
    The blacs array desc wrapper.
    """
    _fields_ = [
        ("dtype", ctypes.c_int),
        ("ctxt", ctypes.c_int),
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("mb", ctypes.c_int),
        ("nb", ctypes.c_int),
        ("rsrc", ctypes.c_int),
        ("csrc", ctypes.c_int),
        ("lld", ctypes.c_int),
    ]

    @property
    def c_dtype(self):
        return ctypes.c_int.from_buffer(self, self.__class__.dtype.offset)

    @property
    def c_ctxt(self):
        return ctypes.c_int.from_buffer(self, self.__class__.ctxt.offset)

    @property
    def c_m(self):
        return ctypes.c_int.from_buffer(self, self.__class__.m.offset)

    @property
    def c_n(self):
        return ctypes.c_int.from_buffer(self, self.__class__.n.offset)

    @property
    def c_mb(self):
        return ctypes.c_int.from_buffer(self, self.__class__.mb.offset)

    @property
    def c_nb(self):
        return ctypes.c_int.from_buffer(self, self.__class__.nb.offset)

    @property
    def c_rsrc(self):
        return ctypes.c_int.from_buffer(self, self.__class__.rsrc.offset)

    @property
    def c_csrc(self):
        return ctypes.c_int.from_buffer(self, self.__class__.csrc.offset)

    @property
    def c_lld(self):
        return ctypes.c_int.from_buffer(self, self.__class__.lld.offset)


class Array(ArrayDesc):
    """
    A blacs array with array desc.
    """

    def __init__(self, context, m, n, mb, nb, *, data=None, dtype=None):
        """
        Create a blacs array in the given Context. User must specify either `data` or `dtype`.

        Parameters
        ----------
        context : Context
            The context where the array created.
        m, n : int
            The row and column of the array.
        mb, nb : int
            The block size on row and column.
        data : np.ndarray, optional
            The created blacs array does not owndata, and just use the data in the given numpy array.
        dtype : np.dtype, optional
            Create an new numpy array as the data of the created blacs array.

        See Also
        --------
        Context.array : Create a blacs array in the Context.
        """
        # dtype is always 1 for dense matrix.
        # rsrc and csrc is always 0.
        # lld will be set layer
        super().__init__(1, context.ictxt, m, n, mb, nb, 0, 0, 0)
        # Record the owner context
        self.context = context
        # Get the local row and column in the current process
        self.local_m = self.context.scalapack.numroc(self.c_m, self.c_mb, context.myrow, Scalapack.zero, context.nprow)
        self.local_n = self.context.scalapack.numroc(self.c_n, self.c_nb, context.mycol, Scalapack.zero, context.npcol)
        # lld is just the row of the local array for the fortran style.
        self.lld = self.local_m

        # Set the local array.
        if data is not None:
            # data and dtype cannot be set at the same time
            if dtype is not None:
                raise RuntimeError("Data and dtype cannot be set at the same time")
            # The given data should have the correct shape
            if context and data.shape != (self.local_m, self.local_n):
                raise RuntimeError(f"Given data local shape mismatch, {data.shape} != {(self.local_m, self.local_n)}")
            # Use the array from the given numpy array.
            self.data = data
            # Check the contiguous
            if self.context.layout.value == b'C':
                # The given numpy array must be fortran contiguous.
                if not self.data.flags.f_contiguous:
                    raise RuntimeError("Scalapack array must be Fortran contiguous")
            else:
                # The given numpy array must be c contiguous.
                if not self.data.flags.c_contiguous:
                    raise RuntimeError("Scalapack array must be C contiguous")
        else:
            order = 'F' if self.context.layout.value == b'C' else 'C'
            if context:
                # Create the local array, by the shape calculated by numroc.
                self.data = np.zeros([self.local_m, self.local_n], dtype=dtype, order=order)
            else:
                # Just a placeholder, the current process is out of the grid.
                self.data = np.zeros([1, 1], dtype=dtype, order=order)

    def col_indxl2g(self, jl):
        """
        Wrapper around scalapack.indxl2g to convert the local index
        into a column into its global index.

        Parameters
        ----------
        jl : int 
            local column index in the Fortran context, so it is 1-based!

        Returns
        -------
        j : int
            global column index in the Fortran context, so it is 1-based!
        """
        j = self.context.scalapack.indxl2g(jl,self.c_nb,self.context.mycol.value,0,self.context.npcol.value)
        return j

    def row_indxl2g(self, il):
        """
        Wrapper around scalapack.indxl2g to convert the local index
        into a row into its global index.

        Parameters
        ----------
        il : int 
            local row index in the Fortran context, so it is 1-based!

        Returns
        -------
        i : int
            global row index in the Fortran context, so it is 1-based!
        """
        i = self.context.scalapack.indxl2g(il,self.c_mb,self.context.myrow.value,0,self.context.nprow.value)
        return i
    
    def pdgemr2d(self,dest):
        """
        A wrapper around scalapack.pdgemr2d. Called on the source of the transfer, and with de destination 
        passed as an argument.
        """
        # we must use the context encompassing all ranks of both contexts. 
        # That should be the one with the most ranks. (a bit tricky, I admit)
        src_context = self.context
        dst_context = dest.context
        src_size = src_context.nprow.value * src_context.npcol.value 
        dst_size = dst_context.nprow.value * dst_context.npcol.value
        context = src_context if src_size > dst_size else dst_context

        self.context.scalapack.pdgemr2d(
            self.c_m.value, self.c_n.value, 
            *self.scalapack_params(),
            *dest.scalapack_params(),
            context.ictxt,
        )

    def pdsyev(self,eigenvalues=None,eigenvectors=None):
        """
        Wrapper for scalapack.pdsyev

        Parameters
        ----------
        eigenvalues : numpy array of length na or None
            eigenvalues will be stored here

        eigenvectors : Distributed 2D array na x na or None
            eigenvectors will be stored here

        Returns
        -------
        tuple : (eigenvalues,eigenvectors,info)
        """
        na = self.c_m.value

        ev = eigenvalues  if not eigenvalues  is None else np.empty(na, dtype=float) 
        EV = eigenvectors if not eigenvectors is None else self.context.array(na, na, self.c_mb.value, self.c_nb.value, dtype=float)
        
        work = np.array([0.0])
        lwork = -1
        info = np.array([1]) # an ordinary Python variable cannot be used as an output argument.

        self.context.scalapack.pdsyev(
            b'V', b'L', na,
            *self.scalapack_params(),
            ev,
            *EV.scalapack_params(),
            work, lwork,
            info
        )
        lwork = int(work[0])
        # print(f"{lwork=}")
        work = np.zeros(lwork,dtype=float,order='F')
        self.context.scalapack.pdsyev(
            b'V', b'L', na,
            *self.scalapack_params(),
            ev,
            *EV.scalapack_params(),
            work, lwork,
            info
        )
        if info[0] != 0:
            print(f"[{self.context.rank.value}/{selfcontext.size.value}] ({self.context.myrow.value},{self.context.mycol.value}): distributed solution did not converge {info=}.", file=sys.stderr)

        return (ev,EV,info)


    def scalapack_params(self):
        """
        Get parameters use to pass to scalapack. Scalapack usually take matrix as parameter with the following format.
        The pointer to data, the row and column indices in thr global matrix indicating the start point of all local matrix,
        and array desc.

        Returns
        -------
        args : tuple
            The arguments to be passed to scalapack function.
        """
        return (
            Scalapack.numpy_ptr(self.data),
            Scalapack.one,
            Scalapack.one,
            self,
        )

    def lapack_params(self):
        """
        Get parameters use to pass to lapack. Lapack usually take matrix as parameter with the following format.
        The pointer to data, and lld.

        Returns
        -------
        args : tuple
            The arguments to be passed to lapack function.
        """
        return (
            Scalapack.numpy_ptr(self.data),
            self.c_lld,
        )


class Scalapack():

    @staticmethod
    def _default_dll_loader(lib):
        """
        The default loader for the dynamic shared library.

        Parameters
        ----------
        lib : str
            The name of the dynamic shared library.

        Returns
        -------
        Any
            The dynamic shared library object, it is usually a ctypes.CDLL.
        """
        return ctypes.CDLL(lib, mode=os.RTLD_LAZY | os.RTLD_GLOBAL)

    def __init__(self, *libs, loader=None):
        """
        Create scalapack library handle from the given pathes as scalapack dynamic linked lbiraries.

        Parameters
        ----------
        *libs : list[str]
            The dynamics linked libraries full pathes or just the file names if in the default directory.
        loader : Callable
            The loader for the dynamic shared libraries.
        """
        if loader is None:
            loader = self._default_dll_loader
        self.libs = [loader(lib) for lib in libs]
        self.function_database = {}

        # Common used functions
        for name in ["p?gemm", "p?gemv", "?gemv", "p?gemr2d"]:
            setattr(self, name.replace("?", ""),
                    {btype: getattr(self, name.replace("?", btype.lower())) for btype in "SDCZ"})
        self.pheevd = {"S": self.pssyevd, "D": self.pdsyevd, "C": self.pcheevd, "Z": self.pzheevd}

    def __call__(self, layout, nprow, npcol):
        """
        Create a blacs Context.

        Parameters
        ----------
        layout : b'R' | b'C'
            The layout of the blacs grid, row major or column major.
        nprow, npcol : int
            The row and column of the blacs grid.

        Returns
        -------
        Context
            The created blacs Context.
        """
        if layout not in [b'R', b'C']:
            raise ValueError(f"layout should be b'R' or b'C' but it is {layout}.")
        return Context(self, layout, nprow, npcol)

    def __getattr__(self, name, suffix='_'):
        """
        Get a function from the scalapack library, and wrap it by fortran function wrapper.

        Parameters
        ----------
        name : str
            The function name, without suffix "_" in fortran function.

        Returns
        -------
            The function wrapped by fortran function wrapper.
        """
        # All function will be cached in function_database
        if _DBG:
            print(f"looking for function {name}")
        if name not in self.function_database:
            # Get the real function name since fortran function has a suffix "_" in its name.
            real_name = name + suffix
            # Look it up in all libraries
            for lib in self.libs:
                # If it is in a library
                if hasattr(lib, real_name):
                    # Get it, wrapper it, save it, and return.
                    self.function_database[name] = self._fortran_function(getattr(lib, real_name))
                    break
            else:
                raise AttributeError(f"No function named '{real_name}' in the libraries")
        return self.function_database[name]

    class Val():
        """
        An auxiliary type to indicate an argument should be passed by value to a fortran function.
        """

        def __init__(self, value):
            """
            Create Val wrapper

            Parameters
            ----------
            value
                The value should be passed by value.
            """
            self.value = value

    @classmethod
    def _resolve_arg(cls, arg):
        """
        Resolve argument as the fortran style: pass by reference by default, except indicated by `Val`.
        """
        # if _DBG:
        #     print(f"_resolve_arg : {type(arg)=} {arg=}") # for debugging
        if isinstance(arg, cls.Val):
            if _DBG:
                print(f"_resolve_arg : {type(arg)=} {arg.value=} cls.Val") # for debugging
            # This argument is specified to pass by value
            return arg.value
        elif isinstance(arg, int):
            if _DBG:
                print(f"_resolve_arg : {type(arg)=} {arg=} int") # for debugging
            # This is a python int, wrap it in c_int
            arg = ctypes.c_int(arg)
            return ctypes.byref(arg)
        elif isinstance(arg, bytes):
            if _DBG:
                print(f"_resolve_arg : {type(arg)=} {arg=} bytes") # for debugging
            # This is a python bytes, wrap it in c_char_p
            arg = ctypes.c_char_p(arg)
            return arg
        elif isinstance(arg, np.ndarray):
            if _DBG:
                print(f"_resolve_arg : {type(arg)=} {arg=} np.ndarray") # for debugging
            return arg.ctypes.data_as(ctypes.c_void_p)
        elif isinstance(arg, ctypes.c_void_p):
            if _DBG:
                print(f"_resolve_arg : {type(arg)=} {arg=} ctypes.c_void_p") # for debugging
            return arg.value
        else:
            if _DBG:
                print(f"_resolve_arg : {type(arg)=} {arg.value=} else") # for debugging
            # This must be already a ctypes object, get the reference.
            return ctypes.byref(arg)

    @classmethod
    def _fortran_function(cls, function):
        """
        Wrapper for fortran function in dynamic linked library.

        See Also
        --------
        Scalapack._resolve_arg : Resolve argument to fit the fortran calling style.
        """

        def result(*args):
            if _DBG:
                print(f"{len(args)} arguments")
            return function(*(cls._resolve_arg(arg) for arg in args))

        result.__doc__ = function.__doc__
        return result

    @classmethod
    def numpy_ptr(cls, array):
        """
        Get the pointer from a numpy array.

        Note: Fortran take argument as reference. But the pointer itself does not need to be a reference.
        So use `cls.Val` to pass it as value.

        Parameters
        ----------
        array : np.ndarray
            The pointer will be got from this numpy array.

        Returns
        -------
        ctypes.c_void_p
            The pointer to the numpy array data.
        """
        return cls.Val(array.ctypes.data_as(ctypes.c_void_p))

    # Some constants in c
    zero = ctypes.c_int(0)
    one = ctypes.c_int(1)
    neg_one = ctypes.c_int(-1)

    s_zero = ctypes.c_float(0)
    s_one = ctypes.c_float(1)
    d_zero = ctypes.c_double(0)
    d_one = ctypes.c_double(1)
    c_zero = (ctypes.c_float * 2)(0, 0)
    c_one = (ctypes.c_float * 2)(1, 0)
    z_zero = (ctypes.c_double * 2)(0, 0)
    z_one = (ctypes.c_double * 2)(1, 0)
    f_zero = {"I": zero, "S": s_zero, "D": d_zero, "C": c_zero, "Z": z_zero}
    f_one = {"I": one, "S": s_one, "D": d_one, "C": c_one, "Z": z_one}

    # Add ctypes as a field of the Scalapack
    import ctypes


sys.modules[__name__] = Scalapack
