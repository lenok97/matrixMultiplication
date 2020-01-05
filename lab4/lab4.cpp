#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
using namespace std;

const int root = 0;

int main(int argc, char **argv)
{
	int size, rank, numworkers, n, source, dest, rows, offset, i, j, k;
	int **a, **b, **c;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;
	numworkers = size - 1;

	/*---------------------------- master ----------------------------*/
	if (rank == root)
	{
		cout << "Matrix multiplication" << endl << "Process count = " << size << endl;
		cout << "Enter n:" << endl;
		cin >> n;
		a = new int*[n];
		b = new int*[n];

		for (i = 0; i < n; i++)
		{
			a[i] = new int[n];
			b[i] = new int[n];
			for (j = 0; j < n; j++)
			{
				a[i][j] = 1;
				b[i][j] = 2;
			}
		}

		rows = n / numworkers;
		offset = 0;
		/* send matrix data to the worker tasks */
		for (dest = 1; dest <= numworkers; dest++)
		{
			MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], rows*n, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
			MPI_Send(&b, n*n, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
			offset = offset + rows;
		}

	}
		
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank !=root) 
	{
		source = 0;
		MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&a, rows*n, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&b, n*n, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
		c = new int*[n];
		/* Matrix multiplication */
		for (k = 0; k < n; k++)
		{
			c[k] = new int[n];
			for (i = 0; i < rows; i++)
			{
				c[i][k] = 0;
				for (j = 0; j < n; j++)
					c[i][k] = c[i][k] + a[i][j] * b[j][k];
			}
		}

		MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
		MPI_Send(&c, rows*n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
	}

	if (rank == root)
	{
		c = new int*[n];
		for (i = 1; i <= numworkers; i++)
		{
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[offset][0], rows*n, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
		}

		printf("Here is the result matrix:\n");
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
				printf("%6.2f   ", c[i][j]);
			printf("\n");
		}

		//fprintf(stdout, "Time = %.6f\n\n",
			//(stop.tv_sec + stop.tv_usec*1e-6) - (start.tv_sec + start.tv_usec*1e-6));
	}

	MPI_Finalize();
}