#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
using namespace std;

const int root = 0;
const int shiftTag = 1;

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

void generateSimpleMatrixWithI(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			matrix[n * i + j] = i;
}

void generateSimpleMatrixWithJ(int n, int m, int* matrix)
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
	for (int i = 0; i < n - 1; i++)
		for (int j = i; j < n; j++)
		{
			temp = inmatrix[n*j + i];
			inmatrix[n*j + i] = inmatrix[n*i + j];
			inmatrix[n*i + j] = temp;
		}
}

int main(int argc, char *argv[])
{
	int rank, size;
	double endTime, startTime;
	bool randomMatrix = false;



	MPI_Status Status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	while (true)
	{
		int n, workPerProc, extraWork, rowsCount, max=0;
		int* matrixA = NULL;
		int* matrixB = NULL;
		int* matrixC = NULL;

		// MPI_Scatterv/MPI_Gatherv params
		int* sendcounts = new int[size]; //количество элементов, принимаемых от каждого процесса
		int* senddispls = new int[size]; //начало расположения элементов блока, посылаемого i-му процессу
		int* recvcounts = new int[size]; //количество элементов, отправляемых каждым процессом
		int* recvdispls = new int[size]; //начало расположения элементов блока, принимаемых от i-ого процесса
		int* start = new int[size];
		int* stop = new int[size];

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
				generateSimpleMatrixWithI(n, n, matrixA);
				generateSimpleMatrixWithJ(n, n, matrixB);
			}
			
			if (n < 100)
			{
				cout << "matrix A:" << endl;
				printMatrix(n, n, matrixA);
				cout << "matrix B:" << endl;
				printMatrix(n, n, matrixB);
			}
			startTime = MPI_Wtime();
			transpMatrix(n, matrixB);

			workPerProc = n / size;
			max = workPerProc*n;
			extraWork = n % size;
			if (extraWork > 0)
				max+=n;
			int totalDispl = 0;
			// Calculate MPI_Scatterv/MPI_Gatherv params
			for (int i = 0; i < size; i++)
			{
				recvcounts[i] = (i < extraWork) ? workPerProc + 1 : workPerProc; 
				sendcounts[i] = recvcounts[i] * n; //Определяем сколько элементов массива войдёт в его часть
				recvdispls[i] = totalDispl; //Определяем начальную строчку передаваемого сообщения
				senddispls[i] = totalDispl * n; //Определяем начальный элемент передаваемого сообщения
				start[i] = totalDispl;
				stop[i] = start[i] + recvcounts[i];
				totalDispl += recvcounts[i]; 
			}

			/*for (int i = 0; i < size; ++i)
				if (recvcounts[i] > max)
					max = recvcounts[i];
*/
			matrixC = new int[n*n];
		}

		MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);
		MPI_Bcast(&max, 1, MPI_INT, root, MPI_COMM_WORLD); 
		MPI_Bcast(start, size, MPI_INT, root, MPI_COMM_WORLD);
		MPI_Bcast(stop, size, MPI_INT, root, MPI_COMM_WORLD);

		// горизонтальное ленточное разбиение
		workPerProc = n / size;
		extraWork = n % size;

		rowsCount = workPerProc;
		if (rank < extraWork)
			rowsCount++;
		int* partA = new int[rowsCount * n];
		int* partB = new int[max * n];
		int* shiftPart= new int[max * n];
		int* partC = new int[rowsCount * n];
		// разбивает сообщение из буфера посылки процесса root на части
		MPI_Scatterv(matrixA, sendcounts, senddispls, MPI_INT, partA, rowsCount * n, MPI_INT, root, MPI_COMM_WORLD);
		MPI_Scatterv(matrixB, sendcounts, senddispls, MPI_INT, partB, max * n, MPI_INT, root, MPI_COMM_WORLD);

		//выполнения сдвига по цепи процессов
		int nextProc, prevProc;
		nextProc = rank + 1;
		if (rank == size - 1)
			nextProc = 0;
		prevProc = rank - 1;
		if (rank == 0)
			prevProc = size - 1;

		for (int p = 0; p < size; p++)
		{
			int local_start = start[(p + rank) % size];
			int local_stop = stop[(p + rank) % size];
			printMatrix(n, workPerProc, partB);
			for (int i = 0; i < rowsCount; i++)
			{
				for (int j = local_start; j < local_stop; j++) 
				{
					partC[i * n + j] = 0;
					for (int k = 0; k < n; k++)
					{
						partC[i * n + j] += partA[i * n + k] * partB[(j - local_start) * n + k];
						cout << rank << ": " << partB[(j - local_start) * n + k]<<endl;
					}
				}
			}
			
			//MPI_Sendrecv_replace(partB, BSize * n, MPI_INT, nextProc, shiftTag, prevProc, shiftTag, MPI_COMM_WORLD, &Status);
			MPI_Sendrecv(partB, max * n, MPI_INT, prevProc, shiftTag, shiftPart, max * n, MPI_INT, nextProc, shiftTag, MPI_COMM_WORLD, &Status);
			for (int t = 0; t < max * n; t++)
				partB[t] = shiftPart[t];

		}

		printMatrix(n, rowsCount, partC);

		// собирает блоки с разным числом элементов от каждого процесса
		MPI_Gatherv(partC, rowsCount*n, MPI_INT, matrixC, sendcounts,senddispls, MPI_INT, root, MPI_COMM_WORLD);

		if (rank == root)
		{
			endTime = MPI_Wtime();
			cout << "matrix C:" << endl;
			printMatrix(n, n, matrixC);
			cout << endl << "Time elapsed: " << (endTime - startTime) * 1000 << "ms\n" << endl;
			//randomMatrix = !randomMatrix;
		}
	}
	MPI_Finalize();
	return 0;
}
