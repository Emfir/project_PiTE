#include "solver.h"
#include <Python.h>
#include <stdio.h>

PyObject *value_error(const char *err) {
  PyErr_SetString(PyExc_ValueError, err);
  return NULL;
}

static PyObject *sudoku_solver(PyObject *self, PyObject *args) {

  long long stab[sud_sz][sud_sz] = {0};
  const long long sudoku_sz = sud_sz * sud_sz;
  PyObject *sudoku_vect;
  if (!PyArg_ParseTuple(args, "O", &sudoku_vect))
    return value_error("Bad argument");
  long long n = PyList_Size(sudoku_vect);

  if (n != sudoku_sz)
    return value_error("Sudoku vector should contain 81 elements");

  for (int i = 0; i < n; i++) {
    PyObject *temp = PyList_GetItem(sudoku_vect, i);
    if (!PyLong_Check(temp))
      return value_error("List element is not an int as it should be");
    set_elem(stab, i, PyLong_AsLong(temp));
  }

  if (!solve(stab))
    value_error("Not a solvable sudoku");

  PyObject *solved_sudoku = PyList_New(0);
  if (!solved_sudoku)
    return value_error("Error creating output list");

  for (int i = 0; i < sudoku_sz; ++i) {
    PyObject *temp = PyLong_FromLong(get_elem(stab, i));
    if (PyList_Append(solved_sudoku, temp)) {
      Py_DECREF(solved_sudoku);
      return value_error("Error adding result to the list");
    }
  }

  return solved_sudoku;
}

static PyMethodDef product_methods[] = {
    {"solve", sudoku_solver, METH_VARARGS,
     "Solve 9x9 sudoku puzzle using backtracking, input vector should contain "
     "81 elements (concatenated rows of sudoku matrix)"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef product_definition = {
    PyModuleDef_HEAD_INIT, "ssolver", "This module supplies a method to solve "
                                      "9x9 sudoku puzzle by a brute-force "
                                      "backtracking algorithm",
    -1, product_methods};

PyMODINIT_FUNC PyInit_ssolver(void) {
  Py_Initialize();
  return PyModule_Create(&product_definition);
}
