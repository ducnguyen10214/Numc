#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int majorRowIdx = row * (mat->cols) + col;
    return mat->data[majorRowIdx];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int majorRowIdx = row * (mat->cols) + col;
    mat->data[majorRowIdx] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (cols <= 0 || rows <= 0) {
      return -1;
    }
    matrix *newMatrix = malloc(sizeof(matrix));
    if (newMatrix == NULL) {
        return -2;
    }
    newMatrix->data = calloc(rows * cols, sizeof(double));
    if (newMatrix->data == NULL) {
        return -2;
    }
    newMatrix->rows = rows;
    newMatrix->cols = cols;
    newMatrix->parent = NULL;
    newMatrix->ref_cnt = 1;
    *mat = newMatrix;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
            return;
        }
    }
    deallocate_matrix(mat->parent);
    if (mat->ref_cnt == 0) {
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows < 0 || cols < 0 || from == NULL) {
        return -1;
    }
    matrix *newMatrix = malloc(sizeof(matrix));
    if (newMatrix == NULL) {
        return -2;
    }
    newMatrix->data = from->data + offset;
    newMatrix->rows = rows;
    newMatrix->cols = cols;
    newMatrix->parent = from;
    from->ref_cnt += 1;
    *mat = newMatrix;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    /*
    for (int i = 0; i < mat->rows; i += 1) {
        for (int j = 0; j < mat->cols; j += 1) {
            set(mat, i, j, val);
        }
    }
    */
    
    int arraySize = mat->rows * mat->cols;
    #pragma omp parallel for
    for (int i = 0; i < arraySize / 4 * 4; i += 4) {
        __m256d fourDoubles = _mm256_set1_pd(0);
        fourDoubles = _mm256_set_pd(val, val, val, val);
        _mm256_storeu_pd(mat->data + i, fourDoubles); 
    }
    // Tail case
    for (int i = arraySize / 4 * 4; i < arraySize; i += 1) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /*
    for (int i = 0; i < mat->rows; i += 1) {
        for (int j = 0; j < mat->cols; j += 1) {
            double val = get(mat, i, j);
            if (val < 0) {
                val = -val;
            }
            set(result, i, j, val);
        }
    }
    */
    
    /*
    int arraySize = result->rows * result->cols;
    __m256d fourDoubles = _mm256_set1_pd(0);
    __m256d fourNegOnes = _mm256_set1_pd(-1);
    __m256d multiplied = _mm256_set1_pd(1);
    for (int i = 0; i < arraySize / 4 * 4; i += 4) {
        fourDoubles = _mm256_loadu_pd(mat->data + i);
        multiplied = _mm256_mul_pd(fourDoubles, fourNegOnes);
        _mm256_storeu_pd(result->data + i, _mm256_max_pd(multiplied, fourDoubles));
    }    
    // Tail case
    for (int i = arraySize / 4 * 4; i < arraySize; i += 1) {
        result->data[i] = fabs(mat->data[i]);
    }
    */
    
    int arraySize = result->rows * result->cols;
    #pragma omp parallel for
    for (int i = 0; i < arraySize / 4 * 4; i += 4) {
        __m256d fourDoubles = _mm256_set1_pd(0);
        __m256d fourNegOnes = _mm256_set1_pd(-1);
        __m256d multiplied = _mm256_set1_pd(1);
        fourDoubles = _mm256_loadu_pd(mat->data + i);
        multiplied = _mm256_mul_pd(fourDoubles, fourNegOnes);
        _mm256_storeu_pd(result->data + i, _mm256_max_pd(multiplied, fourDoubles));
    }    
    // Tail case
    for (int i = arraySize / 4 * 4; i < arraySize; i += 1) {
        result->data[i] = fabs(mat->data[i]);
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    /*
    for (int i = 0; i < mat1->rows; i += 1) {
        for (int j = 0; j < mat1->cols; j += 1) {
            double sum = get(mat1, i, j) + get(mat2, i, j);
            set(result, i, j, sum);
        }
    }
    */
    int arraySize = mat1->rows * mat1->cols;
    #pragma omp parallel for
    for (int i = 0; i < arraySize / 4 * 4; i += 4) {
        __m256d fourDoubles1 = _mm256_set1_pd(0);
        __m256d fourDoubles2 = _mm256_set1_pd(0);
        __m256d fourSum = _mm256_set1_pd(0);
        fourDoubles1 = _mm256_loadu_pd(mat1->data + i);
        fourDoubles2 = _mm256_loadu_pd(mat2->data + i);
        fourSum = _mm256_add_pd(fourDoubles1, fourDoubles2);
        _mm256_storeu_pd(result->data + i, fourSum);
    }
    // Tail case
    for (int i = arraySize / 4 * 4; i < arraySize; i += 1) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    /* Naive Implementation
    for (int r1 = 0; r1 < mat1->rows; r1 += 1) {
        for (int c2 = 0; c2 < mat2->cols; c2 += 1) {
            double dotProduct = 0;
            for (int c1 = 0; c1 < mat1->cols; c1 += 1) {
                double tmp1 = get(mat1, r1, c1);
                double tmp2 = get(mat2, c1, c2);
                dotProduct += (tmp1 * tmp2);
            }
            set(result, r1, c2, dotProduct);
        }
    }
    */
    // transpose matrix2
    matrix *transpose = NULL;
    allocate_matrix(&transpose, mat2->cols, mat2->rows);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mat2->rows; i += 1) {
        for (int j = 0; j < mat2->cols; j += 1) {
            //set(transpose, j, i, get(mat2, i, j));
            transpose->data[j * (transpose->cols) + i] = mat2->data[i * (mat2->cols) + j]; 
        }
    }
    /* Improve by multiplying with tranpose of mat2
    for (int i = 0; i < mat1->rows; i += 1) {
        for (int j = 0; j < transpose->rows; j += 1) {
            double dotProduct = 0;
            for (int k = 0; k < transpose->cols; k += 1) {
                dotProduct += (get(mat1, i, k) * get(transpose, j, k));
            }
            set(result, i, j, dotProduct);
        }
    }
    */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mat1->rows; i += 1) {
        for (int j = 0; j < transpose->rows; j += 1) {
            double dotProduct = 0;
            __m256d four1 = _mm256_set1_pd(0);
            __m256d four2 = _mm256_set1_pd(0);
            __m256d dot = _mm256_set1_pd(0);
            double arr[4];
            for (int k = 0; k < transpose->cols / 4 * 4; k += 4) {
                four1 = _mm256_loadu_pd(mat1->data + i * mat1->cols + k);
                four2 = _mm256_loadu_pd(transpose->data + j * transpose->cols + k);
                dot = _mm256_fmadd_pd(four1, four2, dot);
            }
            _mm256_storeu_pd(arr, dot);
            dotProduct += (arr[0] + arr[1] + arr[2] + arr[3]);
            for (int k = transpose->cols / 4 * 4; k < transpose->cols; k += 1) {
                //dotProduct += (get(mat1, i, k) * get(transpose, j, k));
                dotProduct += (mat1->data[i * (mat1->cols) + k] * transpose->data[j * (transpose->cols) + k]);
            }
            set(result, i, j, dotProduct);
            //result->data[i * (result->cols) + j] = dotProduct;
        }
    } 
    return 0;
}

/* Helper function to create an identity matrix. Return 0 if succcessful. */
int generate_identity(matrix *mat) {
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < mat->rows; r += 1) {
        for (int c = 0; c < mat->cols; c += 1) {
            if (r == c) {
                //set(mat, r, c, 1);
                mat->data[r * (mat->cols) + c] = 1;
            } else {
                //set(mat, r, c, 0);
                mat->data[r * (mat->cols) + c] = 0;
            }
        }
    }
    return 0;
}

/* Helpfer function to deep copy a matrix into another matrix. Return 0 if successful. */
int deep_copy(matrix *to, matrix *from) {
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < from->rows; r += 1) {
        for (int c = 0; c < from->cols; c += 1) {
            //set(to, r, c, get(from, r, c));
            to->data[r * (to->cols) + c] = from->data[r * (from->cols) + c];
        }
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    /*
    if (pow == 0) {
        for (int r = 0; r < mat->rows; r += 1) {
            for (int c = 0; c < mat->cols; c += 1) {
                if (r == c) {
                    set(result, r, c, 1);
                } else {
                    set(result, r, c, 0);
                }
            }
        }
    } else if (pow == 1) {
        for (int r = 0; r < result->rows; r += 1) {
            for (int c = 0; c < result->cols; c += 1) {
                double val = get(mat, r, c);
                set(result, r, c, val);
            }
        }
    } else {
        matrix *clone = NULL;
        allocate_matrix(&clone, mat->rows, mat->cols);
        for (int i = 0; i < pow - 1; i += 1) {
            if (i == 0) {
                mul_matrix(result, mat, mat);
            } else {
                mul_matrix(result, clone, mat);
            }
            deallocate_matrix(clone);
            allocate_matrix(&clone, result->rows, result->cols);
            for (int i = 0; i < result->rows; i += 1) {
                for (int j = 0; j < result->cols; j += 1) {
                    double resultVal = get(result, i, j);
                    set(clone, i, j, resultVal);
                }
            }
        }
        deallocate_matrix(clone);
    }
    */
    if (pow == 0) {
        generate_identity(result);
    } else {
        matrix *y = NULL;
        allocate_matrix(&y, mat->rows, mat->cols);    
        generate_identity(y);
        
        matrix *y_clone = NULL;
        allocate_matrix(&y_clone, y->rows, y->cols);
        deep_copy(y_clone, y);
        
        matrix *x = NULL;
        allocate_matrix(&x, mat->rows, mat->cols);
        deep_copy(x, mat);

        matrix *x_clone = NULL;
        allocate_matrix(&x_clone, mat->rows, mat->cols);
        deep_copy(x_clone, x);

        while (pow > 1) {
            if (pow % 2 == 0) {
                mul_matrix(x, x_clone, x_clone);
                deep_copy(x_clone, x);
                pow = pow / 2;
            } else {
                mul_matrix(y, x, y_clone);
                mul_matrix(x, x_clone, x_clone);
                deep_copy(y_clone, y);
                deep_copy(x_clone, x);
                pow = (pow - 1) / 2;
            }
        } 
        mul_matrix(result, x, y);
        deallocate_matrix(x);
        deallocate_matrix(y);
        deallocate_matrix(x_clone);
        deallocate_matrix(y_clone);
    }
    return 0;
}
