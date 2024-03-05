# pylint: disable=C0114

import os
import subprocess
from setuptools import find_packages, setup
from torch.utils import cpp_extension

include_dir = cpp_extension.include_paths()
golib_include = os.path.join(os.getcwd(), "pipec")
include_dir.append(golib_include)

compile_commands = ["go", "build", "-C", "pipeec",
                    "-buildmode=c-archive", "-o", "libpipeec.a"]
subprocess.run(compile_commands, check=True)

setup(
    name='pypipeec',
    ext_modules=[
        cpp_extension.CppExtension(
         name='pipeec',
         sources=['pypipeec/module.cpp'],
         include_dirs=include_dir,
         language='c++',
         extra_compile_args=['-Ipipeec'],
         extra_link_args=['-Lpipeec', '-lpipeec']
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    packages=find_packages(where="./pypipeec")
)