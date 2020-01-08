#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
using namespace std;

const int root = 0;

void printMatrix(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
			cout << matrix[n * i + j] << '\t';
		cout << endl;
	}
	cout << endl;
}

void generateSimpleMatrix(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			matrix[n * i + j] = i;
}

void generateSimplematrixB(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			matrix[n * i + j] = j;
}

void generateRandomMatrix(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			matrix[n * i + j] = rand() % 100;
}

int main(int argc, char *argv[])
{
	int rank, size;
	double endTime, startTime;
	bool randomMatrix = false;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	while (true)
	{
		int n, workPerProc, extraWork, rowsCount;
		int* matrixA = NULL;
		int* matrixB = NULL;
		int* matrixC = NULL;
		// MPI_Scatterv/MPI_Gatherv params
		int* sendcounts = new int[size]; //количество элементов, принимаемых от каждого процесса
		int* senddispls = new int[size]; //начало расположения элементов блока, посылаемого i-му процессу
		int* recvcounts = new int[size]; //количество элементов, отправляемых каждым процессом
		int* recvdispls = new int[size]; //начало расположения элементов блока, принимаемых от i-ого процесса

		if (rank == root)
		{
			cout << "Matrix multiplication C=A*B" << endl << "Process count = " << size << endl;
			cout << "Enter n:" << endl;
			cin >> n;

			matrixA = new int[n*n];
			matrixB = new int[n*n];

			if (randomMatrix)
			{
				generateRandomMatrix(n, n, matrixA);
				generateRandomMatrix(n, n, matrixB);
			}
			else
			{
				generateSimpleMatrix(n, n, matrixA);
				generateSimplematrixB(n, n, matrixB);
			}
			
			if (n < 100)
			{
				cout << "matrix A:" << endl;
				printMatrix(n, n, matrixA);
				cout << "matrix B:" << endl;
				printMatrix(n, n, matrixB);
			}
			startTime = MPI_Wtime();

			workPerProc = n / size;
			extraWork = n % size;;
			int totalDispl = 0;
			// MPI_Scatterv/MPI_Gatherv params
			for (int i = 0; i < size; i++)
			{
				int t1 = (i < extraWork) ? workPerProc + 1 : workPerProc;
				recvcounts[i] = t1 * n;
				sendcounts[i] = t1 * n;
				int t2 = totalDispl;
				recvdispls[i] = t2 * n;
				senddispls[i] = t2 * n;
				totalDispl += t1;
			}
			matrixC = new int[n*n];
		}

		MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);
		if (rank!=root)
			matrixB=new int[n*n];
		MPI_Bcast(matrixB, n*n, MPI_INT, root, MPI_COMM_WORLD); // это плохо
		// горизонтаальное ленточное разбиение
		workPerProc = n / size;
		extraWork = n % size;
		rowsCount = workPerProc;
		if (rank < extraWork)
			rowsCount++;
		int* partA = new int[rowsCount * n];

		// разбивает сообщение из буфера посылки процесса root на части
		MPI_Scatterv(matrixA, sendcounts, senddispls, MPI_INT, partA, rowsCount * n, MPI_INT, root, MPI_COMM_WORLD);
		
		if (rank == root)
		{
			delete[] sendcounts;
			delete[] senddispls;
			delete[] matrixA;
		}
		
		int* partC = new int[rowsCount*n];
		for (int row = 0; row < rowsCount; row++)
		{
			for (int j = 0; j < n; j++)
			{
				partC[row * n + j] = 0;
				for (int k = 0; k < n; k++)
				{
					partC[row * n + j] += partA[row * n + k] * matrixB[k * n + j];;
				}
			}
		}
		
		// собирает блоки с разным числом элементов от каждого процесса
		MPI_Gatherv(partC, rowsCount*n, MPI_INT, matrixC, recvcounts, recvdispls, MPI_INT, root, MPI_COMM_WORLD);
		delete[] partA;
		delete[] partC;
		delete[] matrixB;

		if (rank == root)
		{
			endTime = MPI_Wtime();
			cout << "matrix C:" << endl;
			printMatrix(n, n, matrixC);
			delete[] recvcounts;
			delete[] recvdispls;
			delete[] matrixC;
			cout << endl << "Time elapsed: " << (endTime - startTime) * 1000 << "ms\n" << endl;
			//randomMatrix = !randomMatrix;
		}
	}
	MPI_Finalize();
	return 0;
}
