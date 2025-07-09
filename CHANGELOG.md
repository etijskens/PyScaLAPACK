**unreleased**

**v0.0.1**
- in `Scalapack._resolve_args(cls,arg))` add case for converting Numpy arrays to Fortran functions. Arguments of type `np.ndarray` can be passed as:

```Python
elif isinstance(arg, np.ndarray):
    return arg.ctypes.data_as(ctypes.c_void_p)
```

**v0.0.0**
- initial commit.

