/* File:     mpi_dijkstra.c
 * Purpose:  Implement Dijkstra's algorithm for solving the single-source
 *           shortest path problem:  find the length of the shortest path
 *           between a specified vertex and all other vertices in a
 *           directed graph.
 *
 * Compile:  mpicc -g -Wall -o mpi_dijkstra mpi_dijkstra.c
 * Run:      mpiexec -n <p> ./mpi_dijkstra "Project1 Test Data/test.txt" "Result/test_dist.txt" "Result/test_path.txt"
 *           "Project1 Test Data/test.txt":  the path of input file
 *           "Result/test_dist.txt":   the path of the file that stores the distance from 0 to each vertex
 *           "Result/test_path.txt":   the path of the file that stores the shortest path from 0 to each vertex
 *
 * Input:    n, the number of vertices in the digraph
 *           mat, the adjacency matrix of the digraph
 *           store in file "Project1 Test Data/test.txt"
 * Output:   the distance and shortest path from 0 to each vertex
 *           store in file "Result/test_dist.txt" and "Result/test_path.txt"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define MAX_STRING 10000
#define INFINITY 1000000

int Read_n(int my_rank, MPI_Comm comm, FILE *f);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Read_matrix(int loc_mat[], int n, int loc_n,
    MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm, FILE *f);
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank);
void Print_matrix(int loc_mat[], int n, int loc_n,
    MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);

int Find_min_dist(int loc_dist[], int loc_known[], int loc_n, int my_rank, MPI_Comm comm);
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_known[], int loc_n, int n, int my_rank, MPI_Comm comm); /* Dijkstra */
void Print_dists(int loc_dist[], int n, int loc_n, int my_rank, MPI_Comm comm, FILE *f);
void Print_paths(int loc_pred[], int n, int loc_n, int my_rank, MPI_Comm comm, FILE *f);

int main(int argc, char* argv[])
{
    int* loc_mat;
    int* loc_dist;
    int* loc_pred;
    int* loc_known;

    clock_t begin, end;

    FILE *fin, *fout1, *fout2;
    char *testfile = argv[1], *distfile = argv[2], *pathfile = argv[3];

    if (!(fin = fopen(testfile, "r"))) printf("open file failed.\n");

    int n, loc_n, p, my_rank;
    MPI_Comm comm;
    MPI_Datatype blk_col_mpi_t;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    n = Read_n(my_rank, comm, fin);
    loc_n = n / p;
    loc_mat = malloc(n * loc_n * sizeof(int));
    loc_dist = malloc(loc_n * sizeof(int));
    loc_pred = malloc(loc_n * sizeof(int));
    loc_known = malloc(loc_n * sizeof(int));


    /* Build the special MPI_Datatype before doing matrix I/O */
    blk_col_mpi_t = Build_blk_col_type(n, loc_n);

    Read_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm, fin);

    begin = clock();
    Dijkstra(loc_mat, loc_dist, loc_pred, loc_known, loc_n, n, my_rank, comm);
    end = clock();

    if (!(fout1 = fopen(distfile, "w"))) printf("open file failed.\n");
    Print_dists(loc_dist, n, loc_n, my_rank, comm, fout1);
    if (!(fout2 = fopen(pathfile, "w"))) printf("open file failed.\n");
    Print_paths(loc_pred, n, loc_n, my_rank, comm, fout2);

    free(loc_mat);
    free(loc_dist);
    free(loc_pred);
    free(loc_known);

    /* When you're done with the MPI_Datatype, free it */
    MPI_Type_free(&blk_col_mpi_t);

    MPI_Finalize();

    printf("Running time: %f\n", (double)(end-begin) / CLOCKS_PER_SEC);

    return 0;
} /* main */

/*---------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Read in the number of rows in the matrix on process 0
 *            and broadcast this value to the other processes
 * In args:   my_rank:  the calling process' rank
 *            comm:  Communicator containing all calling processes
 * Ret val:   n:  the number of rows in the matrix
 */
int Read_n(int my_rank, MPI_Comm comm, FILE *f)
{
    int n;

    if (my_rank == 0) {
        //printf("Enter the number of rows in the matrix:\n");
        fscanf(f, "%d", &n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    return n;
} /* Read_n */

/*---------------------------------------------------------------------
 * Function:  Build_blk_col_type
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            loc_n = n/p:  number cols in the block column
 * Ret val:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype Build_blk_col_type(int n, int loc_n)
{
    MPI_Aint lb, extent;
    MPI_Datatype block_mpi_t;
    MPI_Datatype first_bc_mpi_t;
    MPI_Datatype blk_col_mpi_t;

    MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
    MPI_Type_get_extent(block_mpi_t, &lb, &extent);

    MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);
    MPI_Type_create_resized(first_bc_mpi_t, lb, extent,
        &blk_col_mpi_t);
    MPI_Type_commit(&blk_col_mpi_t);

    MPI_Type_free(&block_mpi_t);
    MPI_Type_free(&first_bc_mpi_t);

    return blk_col_mpi_t;
} /* Build_blk_col_type */

/*---------------------------------------------------------------------
 * Function:  Read_matrix
 * Purpose:   Read in an nxn matrix of ints on process 0, and
 *            distribute it among the processes so that each
 *            process gets a block column with n rows and n/p
 *            columns
 * In args:   n:  the number of rows in the matrix and the submatrices
 *            loc_n = n/p:  the number of columns in the submatrices
 *            blk_col_mpi_t:  the MPI_Datatype used on process 0
 *            my_rank:  the caller's rank in comm
 *            comm:  Communicator consisting of all the processes
 * Out arg:   loc_mat:  the calling process' submatrix (needs to be
 *               allocated by the caller)
 */
void Read_matrix(int loc_mat[], int n, int loc_n,
    MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm, FILE *f)
{
    int *mat = NULL, i, j;

    if (my_rank == 0) {
        //printf("Enter the matrix:\n");
        mat = malloc(n * n * sizeof(int));
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                fscanf(f, "%d", &mat[i * n + j]);
    }

    MPI_Scatter(mat, 1, blk_col_mpi_t,
        loc_mat, n * loc_n, MPI_INT, 0, comm);

    if (my_rank == 0)
        free(mat);
} /* Read_matrix */

/*---------------------------------------------------------------------
 * Function:  Print_local_matrix
 * Purpose:   Store a process' submatrix as a string and print the
 *            string.  Printing as a string reduces the chance
 *            that another process' output will interrupt the output.
 *            from the calling process.
 * In args:   loc_mat:  the calling process' submatrix
 *            n:  the number of rows in the submatrix
 *            loc_n:  the number of cols in the submatrix
 *            my_rank:  the calling process' rank
 */
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank)
{
    char temp[MAX_STRING];
    char* cp = temp;
    int i, j;

    sprintf(cp, "\n(Print_loc_mat) Proc %d >\n", my_rank);
    cp = temp + strlen(temp);
    for (i = 0; i < n; i++) {
        for (j = 0; j < loc_n; j++) {
            if (loc_mat[i * loc_n + j] == INFINITY)
                sprintf(cp, " i ");
            else
                sprintf(cp, "%2d ", loc_mat[i * loc_n + j]);
            cp = temp + strlen(temp);
        }
        sprintf(cp, "\n");
        cp = temp + strlen(temp);
    }

    printf("%s", temp);
} /* Print_local_matrix */

/*---------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print the matrix that's been distributed among the
 *            processes.
 * In args:   loc_mat:  the calling process' submatrix
 *            n:  number of rows in the matrix and the submatrices
 *            loc_n:  the number of cols in the submatrix
 *            blk_col_mpi_t:  MPI_Datatype used on process 0 to
 *               receive a process' submatrix
 *            my_rank:  the calling process' rank
 *            comm:  Communicator consisting of all the processes
 */
void Print_matrix(int loc_mat[], int n, int loc_n,
    MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm)
{
    int *mat = NULL, i, j;

    if (my_rank == 0)
        mat = malloc(n * n * sizeof(int));
    MPI_Gather(loc_mat, n * loc_n, MPI_INT,
        mat, 1, blk_col_mpi_t, 0, comm);
    if (my_rank == 0) {
        printf("\n(Print_gol_mat) Proc %d>\n", my_rank);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
                if (mat[i * n + j] == INFINITY)
                    printf(" i ");
                else
                    printf("%2d ", mat[i * n + j]);
            printf("\n");
        }
        free(mat);
    }
} /* Print_matrix */

/*-------------------------------------------------------------------
 * Function:    Find_min_dist
 * Purpose:     Find the vertex u with minimum distance to 0
 *              (dist[u]) among the vertices whose distance
 *              to 0 is not known.
 * In args:     dist:  dist[v] = current estimate of distance
 *                 0->v
 *              known:  whether the minimum distance 0-> is
 *                 known
 *              n:  the total number of vertices
 * Ret val:     The vertex u whose distance to 0, dist[u]
 *              is a minimum among vertices whose distance
 *              to 0 is not known.
 */
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n, int my_rank, MPI_Comm comm)
{
    int loc_u = INFINITY;
    int loc_min_dist = INFINITY;
    int loc_v;

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        if (!loc_known[loc_v]) {
            if (loc_dist[loc_v] < loc_min_dist) {
                loc_u = loc_v;
                loc_min_dist = loc_dist[loc_v];
            }
        }
    }
    return loc_u;
} /* Find_min_dist */

/*-------------------------------------------------------------------
 * Function:    Dijkstra
 * Purpose:     Apply Dijkstra's algorithm to the matrix mat
 * In args:     n:  the number of vertices
 *              mat:  adjacency matrix for the graph
 * Out args:    dist:  dist[v] = distance 0 to v.
 *              pred:  pred[v] = predecessor of v on a
 *                  shortest path 0->v.
 */
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_known[], int loc_n, int n, int my_rank, MPI_Comm comm)
{
    int i, global_u, loc_u, loc_v, new_dist;

   /* known[v] = true, if the shortest path 0->v is known */
   /* known[v] = false, otherwise  */

   /* Initialize d and p */
    if (my_rank == 0) {
        loc_dist[0] = 0;
        loc_known[0] = 1;
    }
    else {
        loc_dist[0] = loc_mat[0];
        loc_known[0] = 0;
    }
    loc_pred[0] = 0;

    int v;
    for (v = 1; v < loc_n; v++) {
        loc_dist[v] = loc_mat[v];
        loc_pred[v] = 0;
        loc_known[v] = 0;
    }
    /* On each pass find an additional vertex */
    /* whose distance to 0 is known           */
    for (i = 1; i < n; i++) {
        loc_u = Find_min_dist(loc_dist, loc_known, loc_n, my_rank, comm);

        int my_min[2];
        int glbl_min[2];
        if (loc_u < INFINITY) {
            my_min[0] = loc_dist[loc_u];
            my_min[1] = loc_u + my_rank * loc_n;;
        }
        else {
            my_min[0] = INFINITY;
            my_min[1] = INFINITY;
        }
        MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);
        global_u = glbl_min[1];

        if (global_u / loc_n == my_rank) {
            loc_u = global_u % loc_n;
            loc_known[loc_u] = 1;
        }
        for (loc_v = 0; loc_v < loc_n; loc_v++) {
            if (!loc_known[loc_v]) {
                new_dist = glbl_min[0] + loc_mat[global_u * loc_n + loc_v];
                if (new_dist < loc_dist[loc_v]) {
                    loc_dist[loc_v] = new_dist;
                    loc_pred[loc_v] = global_u;
                }
            }

        } /* for i */
    }
} /* Dijkstra */

/*-------------------------------------------------------------------
 * Function:    Print_dists
 * Purpose:     Print the length of the shortest path from 0 to each
 *              vertex
 * In args:     n:  the number of vertices
 *              dist:  distances from 0 to each vertex v:  dist[v]
 *                 is the length of the shortest path 0->v
 */
void Print_dists(int loc_dist[], int n, int loc_n, int my_rank, MPI_Comm comm, FILE *f)
{

    int v;
    int* dist = NULL;
    if (my_rank == 0) {
        dist = malloc(n * sizeof(int));
    }
    MPI_Gather(loc_dist, loc_n, MPI_INT, dist, loc_n, MPI_INT, 0, comm);

    if (my_rank == 0) {
        printf("The distance from 0 to each vertex is:\n");
        printf("  v    dist 0->v\n");
        printf("----   ---------\n");

        for (v = 1; v < n; v++) {
            printf("%3d       %4d\n", v, dist[v]);
        }
        printf("\n");

        fprintf(f, "The distance from 0 to each vertex is:\n");
        fprintf(f, "  v    dist 0->v\n");
        fprintf(f, "----   ---------\n");

        for (v = 1; v < n; v++) {
            fprintf(f, "%3d       %4d\n", v, dist[v]);
        }
        fprintf(f, "\n");

        free(dist);
    }

} /* Print_dists */

/*-------------------------------------------------------------------
 * Function:    Print_paths
 * Purpose:     Print the shortest path from 0 to each vertex
 * In args:     n:  the number of vertices
 *              pred:  list of predecessors:  pred[v] = u if
 *                 u precedes v on the shortest path 0->v
 */
void Print_paths(int loc_pred[], int n, int loc_n, int my_rank, MPI_Comm comm, FILE *f)
{
    int v, w, *path, count, i;
    int* pred = NULL;
    if (my_rank == 0) {
        pred = malloc(n * sizeof(int));
    }
    MPI_Gather(loc_pred, loc_n, MPI_INT, pred, loc_n, MPI_INT, 0, comm);
    if (my_rank == 0) {
        printf("The shortest path from 0 to each vertex is:\n");

        path = malloc(n * sizeof(int));

        printf("  v     Path 0->v\n");
        printf("----    ---------\n");
        for (v = 1; v < n; v++) {
            printf("%3d:    ", v);
            count = 0;
            w = v;
            while (w != 0) {
                path[count] = w;
                count++;
                w = pred[w];
            }
            printf("0 ");
            for (i = count - 1; i >= 0; i--) {
                printf("%d ", path[i]);
            }
            printf("\n");
        }

        fprintf(f, "The shortest path from 0 to each vertex is:\n");

        path = malloc(n * sizeof(int));

        fprintf(f, "  v     Path 0->v\n");
        fprintf(f, "----    ---------\n");
        for (v = 1; v < n; v++) {
            fprintf(f, "%3d:    ", v);
            count = 0;
            w = v;
            while (w != 0) {
                path[count] = w;
                count++;
                w = pred[w];
            }
            fprintf(f, "0 ");
            for (i = count - 1; i >= 0; i--) {
                fprintf(f, "%d ", path[i]);
            }
            fprintf(f, "\n");
        }

        free(path);
        free(pred);
    }
} /* Print_paths */
