#!/usr/bin/env python3
# encoding: utf-8

from distutils.core import setup, Extension

sfc_module = Extension('ssolver', sources = ['sudoku_solver.c','solver.c'], extra_compile_args=["-O3"], )

setup(name='ssolver',
      version='0.1.0',
      description='Module solves 9x9 sudoku supplied as a vector using backtracking',
      ext_modules=[sfc_module])
