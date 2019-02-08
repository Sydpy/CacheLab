/**
 * A code skeleton for the matrix multiply bonus assignment.
 * 
 * Course: Advanced Computer Architecture, Uppsala University
 * Course Part: Lab assignment 1
 *
 * Author: Andreas Sandberg <andreas.sandberg@it.uu.se>
 *
 * $Id: multiply.c 81 2012-09-13 08:01:46Z andse541 $
 */

#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Size of the matrices to multiply */
#define SIZE 500

/* HINT: The Makefile allows you to specify L1 and L2 block sizes as
 * compile time options.These may be specified when calling make,
 * e.g. "make L1_BLOCK_SIZE=256 L2_BLOCK_SIZE=1024". If the block
 * sizes have been specified on the command line, makefile defines the
 * macros L1_BLOCK_SIZE and L2_BLOCK_SIZE. If you decide to use them,
 * you should setup defaults here if they are undefined.
 */
#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE 256
#endif
#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 1024
#endif


static double mat_a[SIZE][SIZE];
static double mat_b[SIZE][SIZE];
static double mat_c[SIZE][SIZE];
static double mat_ref[SIZE][SIZE];

/* To optimize iterations over the volumns of the b matrix */
static double mat_b_transpose[SIZE][SIZE];

/* Some helpers */
static inline int next_power_of_2(int n) {
        int power_of_2 = n - 1;
        for (int i = 1; i < sizeof(int); i*=2)
                power_of_2 |= power_of_2 >> i;

        return power_of_2 + 1;
}
static inline int min(int a, int b) {
        return a < b ? a : b;
}


/*
 * This function assume that `size` is a power of 2.
 * However, it handles the case where we write outside of the
 * result matrix (i.e, SIZE is not a power of 2)
 *
 * https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Shared-memory_parallelism
 */
static void
_matmul_opt_rec(int a_row, int a_col, int b_row, int b_col, int size) {
        
        int i, j, k;
        int rows, cols, mults;
        int newsize;

        /* If the current submatrix rows can fit inside a L1 block
         * then we can compute the product */
        if (size <= (L1_BLOCK_SIZE / sizeof(double))) {

                /* we need to update the ranges because not every
                 * matrix has a size like 2**n */
                rows = min(SIZE - a_row, size);
                cols = min(SIZE - b_col, size);
                /* number of multiplications per elements
                 * should also be equal to min(SIZE - b_row, size)
                 * as we use square matrices */
                mults = min(SIZE - a_col, size);

                for (j = b_col; j < b_col + cols; j++) {
                        for (i = a_row; i < a_row + rows; i++) {
                                for (k = 0; k < mults; k++) {
                                        mat_c[i][j] += mat_a[i][a_col + k] * mat_b_transpose[j][b_row + k];
                                }
                        }
                }

                return;
        }

        newsize = size/2;

        // A11 (*) B11
        _matmul_opt_rec(a_row, a_col, b_row, b_col, newsize);
        // A11 (*) B12
        _matmul_opt_rec(a_row, a_col + newsize, b_row + newsize, b_col, newsize);
        // A12 (*) B21
        _matmul_opt_rec(a_row, a_col, b_row, b_col + newsize, newsize);
        // A12 (*) B22
        _matmul_opt_rec(a_row, a_col + newsize, b_row + newsize, b_col + newsize, newsize);
        // A21 (*) B11
        _matmul_opt_rec(a_row + newsize, a_col, b_row, b_col, newsize);
        // A22 (*) B21
        _matmul_opt_rec(a_row + newsize, a_col + newsize, b_row + newsize, b_col, newsize);
        // A21 (*) B12
        _matmul_opt_rec(a_row + newsize, a_col, b_row, b_col + newsize, newsize);
        // A22 (*) B22
        _matmul_opt_rec(a_row + newsize, a_col + newsize, b_row + newsize, b_col + newsize, newsize);
}

static void
matmul_opt()
{
        int i, j, k;

        for (j = 0; j < SIZE; j++) {
                for (i = 0; i < SIZE; i++) {
                        mat_b_transpose[i][j] = mat_b[j][i];
                }
        }

        /* Divide and conquer with optimized iterations at the end */
        //_matmul_opt_rec(0, 0, 0, 0, next_power_of_2(SIZE));

        /* Iterative but optimized */
         
        for (j = 0; j < SIZE; j++) {
                for (i = 0; i < SIZE; i++) {
                        for (k = 0; k < SIZE; k++) {
                                mat_c[i][j] += mat_a[i][k] * mat_b_transpose[j][k];
                        }
                }
        }
}

/**
 * Reference implementation of the matrix multiply algorithm. Used to
 * verify the answer from matmul_opt. Do NOT change this function.
 */
static void
matmul_ref()
{
        int i, j, k;

        for (j = 0; j < SIZE; j++) {
                for (i = 0; i < SIZE; i++) {
                        for (k = 0; k < SIZE; k++) {
                                mat_ref[i][j] += mat_a[i][k] * mat_b[k][j];
                        }
                }
        }
}

/**
 * Function used to verify the result. No need to change this one.
 */
static int
verify_result()
{
        double e_sum;
        int i, j;

        e_sum = 0;
        for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++) {
                        e_sum += mat_c[i][j] < mat_ref[i][j] ?
                                mat_ref[i][j] - mat_c[i][j] :
                                mat_c[i][j] - mat_ref[i][j];
                }
        }

        return e_sum < 1E-6;
}

/**
 * Get the time in seconds since some arbitrary point. Used for high
 * precision timing measurements.
 */
static double
get_time()
{
        struct timeval tv;

        if (gettimeofday(&tv, NULL)) {
                fprintf(stderr, "gettimeofday failed. Aborting.\n");
                abort();
        }
        return tv.tv_sec + tv.tv_usec * 1E-6;
}

/**
 * Initialize mat_a and mat_b with "random" data. Write to every
 * element in mat_c to make sure that the kernel allocates physical
 * memory to every page in the matrix before we start doing
 * benchmarking.
 */
        static void
init_matrices()
{
        int i, j;

        for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++) {
                        mat_a[i][j] = ((i + j) & 0x0F) * 0x1P-4;
                        mat_b[i][j] = (((i << 1) + (j >> 1)) & 0x0F) * 0x1P-4;
                }
        }

        memset(mat_c, 0, sizeof(mat_c));
        memset(mat_ref, 0, sizeof(mat_ref));
}

        static void
run_multiply(int verify)
{
        double time_start, time_stop;

        time_start = get_time();

        /* mat_c = mat_a * mat_b */
        matmul_opt();

        time_stop = get_time();
        printf("Time: %.4f\n", time_stop - time_start);

        if (verify) {
                printf("Verifying solution... ");
                time_start = get_time();
                matmul_ref();
                time_stop = get_time();

                if (verify_result())
                        printf("OK\n");
                else
                        printf("MISMATCH\n");

                printf("Reference runtime: %f\n", time_stop - time_start);
        }
}

        static void
usage(FILE *out, const char *argv0)
{
        fprintf(out,
                        "Usage: %s [OPTION]...\n"
                        "\n"
                        "Options:\n"
                        "\t-v\tVerify solution\n"
                        "\t-h\tDisplay usage\n",
                        argv0);
}

        int
main(int argc, char *argv[])
{
        int c;
        int errexit;
        int verify;
        extern char *optarg;
        extern int optind, optopt, opterr;

        errexit = 0;
        verify = 0;
        while ((c = getopt(argc, argv, "vh")) != -1) {
                switch (c) {
                        case 'v':
                                verify = 1;
                                break;
                        case 'h':
                                usage(stdout, argv[0]);
                                exit(0);
                                break;
                        case ':':
                                fprintf(stderr, "%s: option -%c requries an operand\n",
                                                argv[0], optopt);
                                errexit = 1;
                                break;
                        case '?':
                                fprintf(stderr, "%s: illegal option -- %c\n",
                                                argv[0], optopt);
                                errexit = 1;
                                break;
                        default:
                                abort();
                }
        }

        if (errexit) {
                usage(stderr, argv[0]);
                exit(2);
        }

        /* Initialize the matrices with some "random" data. */
        init_matrices();

        run_multiply(verify);

        return 0;
}


/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * indent-tabs-mode: nil
 * c-file-style: "linux"
 * compile-command: "make -k"
 * End:
 */
