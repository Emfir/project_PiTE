#include "solver.h"
#include <stdio.h>

int solve_recursive(long long (*stab)[sud_sz], int i, int j);
int move_ok(long long (*stab)[sud_sz], int i, int j, long long v);

void print_stab(long long (*stab)[sud_sz]) {
  for (int i = 0; i < sud_sz; ++i) {
    for (int j = 0; j < sud_sz; ++j) {
      printf("%lld ", stab[i][j]);
    }
    printf("\n");
  }
}

void set_elem(long long (*stab)[sud_sz], int i, long long elem) {
  int l = i % sud_sz;
  int k = i / sud_sz;
  stab[k][l] = elem;
}

long long get_elem(long long (*stab)[sud_sz], int i) {
  int l = i % sud_sz;
  int k = i / sud_sz;
  return stab[k][l];
}

int check(long long (*stab)[sud_sz], int out) {
  if (out) {
    for (int k = 0; k < sud_sz; ++k)
      for (int l = 0; l < sud_sz; ++l) {
        long long temp = stab[k][l];
        stab[k][l] = 0;
        if (!move_ok(stab, k, l, temp))
          return 0;
        stab[k][l] = temp;
      }
  }
  return out;
}


int solve(long long (*stab)[sud_sz]) { // return 0 on failure

  return check(stab, solve_recursive(stab, 0, 0));
}

int row_ok(long long (*stab)[sud_sz], int i, long long v) {
  for (int k = 0; k < sud_sz; ++k)
    if (stab[i][k] == v)
      return 0;
  return 1;
}

int col_ok(long long (*stab)[sud_sz], int j, long long v) {
  for (int k = 0; k < sud_sz; ++k)
    if (stab[k][j] == v)
      return 0;
  return 1;
}

int sqr_ok(long long (*stab)[sud_sz], int i, int j, long long v) {
  const int sud_sz_root = 3;
  int r = i / sud_sz_root;
  int c = j / sud_sz_root;
  r *= sud_sz_root;
  c *= sud_sz_root;
  for (int k = r; k < r + sud_sz_root; ++k)
    for (int l = c; l < c + sud_sz_root; ++l)
      if (stab[k][l] == v)
        return 0;
  return 1;
}

int move_ok(long long (*stab)[sud_sz], int i, int j, long long v) {
  return row_ok(stab, i, v) && col_ok(stab, j, v) && sqr_ok(stab, i, j, v);
}

int next_i(int i, int j) { // must be always called before next_j
  return j == sud_sz - 1 ? ++i : i;
}

int next_j(int i, int j) { return ++j % sud_sz; }

int solve_recursive(long long (*stab)[sud_sz], int i,
                    int j) { // return 0 on failure
  if (i >= sud_sz /* || j>= sud_sz is always 0, see next_j*/)
    return 1;
  if (stab[i][j] != 0)
    return solve_recursive(stab, next_i(i, j), next_j(i, j));

  for (int v = 1; v < 10; ++v) {
    if (move_ok(stab, i, j, v)) {
      stab[i][j] = v;
      if (solve_recursive(stab, next_i(i, j), next_j(i, j)))
        return 1;
      stab[i][j] = 0;
    }
  }

  return 0;
}

#ifdef TEST

int main(void) {

  /*long long stab[sud_sz][sud_sz] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}
                    };*/

  /*long long stab[sud_sz][sud_sz] = {  {0, 0, 3, 2, 0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0, 0, 8, 0, 7},
                       {6, 0, 0, 9, 0, 0, 5, 3, 0},
                       {0, 0, 9, 0, 4, 0, 2, 0, 0},
                       {0, 7, 5, 0, 0, 2, 0, 9, 0},
                       {0, 0, 0, 0, 0, 0, 0, 5, 0},
                       {4, 0, 1, 0, 0, 0, 0, 0, 0},
                       {0, 8, 0, 5, 6, 1, 0, 0, 0},
                       {7, 0, 0, 0, 0, 0, 0, 0, 0}
                     };*/

  long long stab[sud_sz][sud_sz] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 3, 0, 8, 5},
      {0, 0, 1, 0, 2, 0, 0, 0, 0}, {0, 0, 0, 5, 0, 7, 0, 0, 0},
      {0, 0, 4, 0, 0, 0, 1, 0, 0}, {0, 9, 0, 0, 0, 0, 0, 0, 0},
      {5, 0, 0, 0, 0, 0, 0, 7, 3}, {0, 0, 2, 0, 1, 0, 0, 0, 0},
      {0, 0, 0, 0, 4, 0, 0, 0, 9}};

  /*Py l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 8, 5, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 7, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 7, 3, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 9]*/

  int out = solve(stab);
  printf("solved: %d\n", out);
  print_stab(stab);

  return 0;
}

#endif

