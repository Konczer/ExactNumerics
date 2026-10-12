import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# BOOST_ROOT must contain the boost/ header directory; no Boost libraries needed.
boost_root = os.environ.get("BOOST_ROOT")
setup(
    py_modules=["qmboot", "qmb_fast"],
    ext_modules=[Pybind11Extension(
        "_qmb_native", ["native/bootstrap.cpp"],
        include_dirs=[boost_root] if boost_root else [],
        cxx_std=17,
    ), Pybind11Extension(
        "_qmb_numeric_v2", ["native/numeric.cpp"],
        include_dirs=[boost_root] if boost_root else [],
        cxx_std=17,
    )],
    cmdclass={"build_ext": build_ext},
)
