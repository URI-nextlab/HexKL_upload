#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "hexkl_macro.h"

#define N_ROW   (32)
#define N_COL   (32)
#define N_INNER (32)

int main(void) {
  int res = 0;
  _Float16 *A = NULL, *X = NULL, *W = NULL;
  size_t A_size = N_ROW * N_COL * sizeof(_Float16);
  size_t X_size = N_ROW * N_INNER * sizeof(_Float16);
  size_t W_size = N_COL * N_INNER * sizeof(_Float16);

  A = malloc(A_size);
  X = malloc(X_size);
  W = malloc(W_size);
  if (!A || !X || !W) {
    printf("alloc failed\n");
    return 1;
  }

  // simple init
  for (int i = 0; i < N_ROW * N_INNER; ++i) X[i] = (_Float16)1.0f;
  for (int i = 0; i < N_COL * N_INNER; ++i) W[i] = (_Float16)1.0f;
  for (int i = 0; i < N_ROW * N_COL; ++i) A[i] = (_Float16)0.0f;

  res = hexkl_macro_initialize();
  if (res != AEE_SUCCESS) {
    printf("hexkl_macro_initialize failed: 0x%x\n", res);
    goto out;
  }

  res = hexkl_macro_lock_hmx();
  if (res != AEE_SUCCESS) {
    printf("hexkl_macro_lock_hmx failed: 0x%x\n", res);
    goto fini;
  }

  // prepare layouts expected by Macro API
  hexkl_macro_rm_to_wh_f16_inplace(N_COL, N_INNER, W);
  hexkl_macro_rm_to_ah_f16_inplace(N_ROW, N_INNER, X);

  // perform matmul
  res = hexkl_macro_mm_f16(N_ROW, N_COL, N_INNER, A, X, W);
  if (res != AEE_SUCCESS) {
    printf("hexkl_macro_mm_f16 failed: 0x%x\n", res);
  } else {
    printf("hexkl_macro_mm_f16 completed\n");
  }

  // convert back
  hexkl_macro_ah_to_rm_f16_inplace(N_ROW, N_COL, A);

  // print first element
  printf("A[0]=%f\n", (float)A[0]);

  res = hexkl_macro_unlock_hmx();
  if (res != AEE_SUCCESS) printf("hexkl_macro_unlock_hmx failed: 0x%x\n", res);

fini:
  hexkl_macro_finalize();
out:
  if (A) free(A);
  if (X) free(X);
  if (W) free(W);
  return (res == AEE_SUCCESS) ? 0 : 1;
}
