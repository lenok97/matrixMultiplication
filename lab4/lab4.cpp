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
//void printTransponentMatrix(int n, int m, int* matrix)
//{
//	for (int j = 0; j < n; j++)
//	{
//		for (int i = 0; i < m; i++)
//			cout << matrix[n * i + j] << '\t';
//		cout << endl;
//	}
//	cout << endl;
//}

void generateSimpleMatrix(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			matrix[n * i + j] = i;
}

void generateSimpleMatrix2(int n, int m, int* matrix)
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

void transpMatrix(int n, int* inmatrix)
{
	int temp;
	for (int i = 0; i < n-1; i++)
		for (int j=i; j<n; j++)
		{
			temp = inmatrix[n*j+i];
			inmatrix[n*j + i] = inmatrix[n*i + j];
			inmatrix[n*i + j] = temp;
		}
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
		int* matrix1 = NULL;
		int* matrix2 = NULL;
		int* result = NULL;
		// MPI_Scatterv/MPI_Gatherv params
		int* sendcounts = new int[size]; //количество элементов, принимаемых от каждого процесса
		int* senddispls = new int[size]; //начало расположения элементов блока, посылаемого i-му процессу
		int* recvcounts = new int[size]; //количество элементов, отправляемых каждым процессом
		int* recvdispls = new int[size]; //начало расположения элементов блока, принимаемых от i-ого процесса

		if (rank == root)
		{
			cout << "Matrix multiplication" << endl << "Process count = " << size << endl;
			cout << "Enter n:" << endl;
			cin >> n;

			matrix1 = new int[n*n];
			matrix2 = new int[n*n];

			if (randomMatrix)
			{
				generateRandomMatrix(n, n, matrix1);
				generateRandomMatrix(n, n, matrix2);
			}
			else
			{
				generateSimpleMatrix(n, n, matrix1);
				generateSimpleMatrix2(n, n, matrix2);
			}
			
			if (n < 100)
			{
				cout << "matrix 1:" << endl;
				printMatrix(n, n, matrix1);
				cout << "matrix 2:" << endl;
				printMatrix(n, n, matrix2);
			}
			startTime = MPI_Wtime();
			transpMatrix(n, matrix2);
			cout << "trans matrix 2:" << endl;
			printMatrix(n, n, matrix2);

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
			result = new int[n*n];
		}

		MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// горизонтаальное ленточное разбиение
		workPerProc = n / size;
		extraWork = n % size;
		rowsCount = workPerProc;
		if (rank < extraWork)
			rowsCount++;
		int* matrixPart1 = new int[rowsCount * n];
		int* matrixPart2 = new int[rowsCount * n];
		// разбивает сообщение из буфера посылки процесса root на части
		MPI_Scatterv(matrix1, sendcounts, senddispls, MPI_INT, matrixPart1, rowsCount * n, MPI_INT, root, MPI_COMM_WORLD);
		MPI_Scatterv(matrix2, sendcounts, senddispls, MPI_INT, matrixPart2, rowsCount * n, MPI_INT, root, MPI_COMM_WORLD);
		if (rank == root)
		{
			delete[] sendcounts;
			delete[] senddispls;
			delete[] matrix1;
			delete[] matrix2;
		}
		cout << rank << ": m1 ";
		printMatrix(n, rowsCount, matrixPart1);
		cout << rank << ": m2 ";
		printMatrix(n, rowsCount, matrixPart2);
		int* tempResult = new int[rowsCount*n];
		for (int row = 0; row < rowsCount; row++)
		{
			for (int j = 0; j < n; j++)
			{
				tempResult[row * n + j] = 0;
				for (int k = 0; k < n; k++)
				{
					//tempResult[i] += matrixPart[i *n + j] * vector[j];
					tempResult[row * n + j] += matrixPart1[row * n + k] * matrixPart2[row * n + j];
					cout <<rank<<": "<< matrixPart1[row * n + k]<<" * "<< matrixPart2[row * n + j] << endl;
				}
			}
		}
		printMatrix(n, rowsCount, tempResult);
		MPI_Barrier(MPI_COMM_WORLD);
		// собирает блоки с разным числом элементов от каждого процесса
		MPI_Gatherv(tempResult, rowsCount*n, MPI_INT, result, recvcounts, recvdispls, MPI_INT, root, MPI_COMM_WORLD);
		delete[] matrixPart1;
		delete[] matrixPart2;
		delete[] tempResult;

		if (rank == root)
		{
			endTime = MPI_Wtime();
			cout << "result:" << endl;
			printMatrix(n, n, result);
			delete[] recvcounts;
			delete[] recvdispls;
			delete[] result;
			cout << endl << "Time elapsed: " << (endTime - startTime) * 1000 << "ms\n" << endl;
			//randomMatrix = !randomMatrix;
		}
	}
	MPI_Finalize();
	return 0;
}
