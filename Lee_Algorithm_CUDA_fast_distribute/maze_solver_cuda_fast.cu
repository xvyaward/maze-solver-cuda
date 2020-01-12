// ***************************************************
// writer : changhun lee
// you need openCV 4.1.2-vc14_vc15 to launch this code
// ***************************************************

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <queue>

#include <stdlib.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_VEC_SIZE 2E4
#define Q_TAIL_INDEX 2E3

using namespace cv;
using namespace std;

int length;

struct POINT {
	int x;
	int y;
};

void lee_algorithm(POINT start, POINT end, int** IMGMAT, int length);
void backtrace(POINT start, POINT end, int** IMGMAT, int** BTRACE, int length);
__global__ void lee_algorithm_cuda(POINT* C, int* C_size, POINT* end_gpu, int* IMGMAT_endP_gpu, int* IMGMAT, int* legnth);
__global__ void backtrace_cuda(int* IMGMAT_constant, int* BTRACE_min, int* length);

int main() {
	clock_t start_t, end_t;
	double runtime;

	Mat maze_image;
	maze_image = imread("input_image.png", IMREAD_GRAYSCALE);
	length = maze_image.cols;

	namedWindow("Maze Image");
	imshow("Maze Image", maze_image);
	printf("Press any key to continue\n");

	waitKey(0);

	start_t = clock();

	int height = maze_image.rows;
	int width = maze_image.cols;
	uchar* image_data = maze_image.data;

	int** IMGMAT = new int * [height];
	IMGMAT[0] = new int[height * width];
	for (int i = 1; i < height; i++) {
		IMGMAT[i] = IMGMAT[i - 1] + width;
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			IMGMAT[i][j] = 0;
		}
	}

	int** BTRACE;
	BTRACE = new int* [height];
	for (int i = 0; i < height; i++) {
		BTRACE[i] = new int[width];
		memset(BTRACE[i], 0, sizeof(int) * width);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (image_data[i * height + j] == 0) IMGMAT[i][j] = -1;
		}
	}

	POINT start;
	POINT end;
	for (int i = 0; i < width; i++) {
		if (IMGMAT[0][i] != -1) {
			start = { 0, i }; 
			break;
		}
	}
	for (int i = 0; i < width; i++) {
		if (IMGMAT[height - 1][i] != -1) {
			end = { height - 1, i };
			break;
		}
	}

	clock_t start_tt, end_tt;
	double runtime1;

	start_tt = clock();

	lee_algorithm(start, end, IMGMAT, length);
	backtrace(start, end, IMGMAT, BTRACE, length);

	end_tt = clock();

	runtime1 = (double)(end_tt - start_tt);
	printf("runtime : %f ms\n", runtime1);

	Mat maze_solved_image(height, width, CV_8UC1);
	uchar* image_solved_data = maze_solved_image.data;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			image_solved_data[i * height + j] = 255 - (IMGMAT[i][j] * 255 / IMGMAT[end.x][end.y]);
		}
	}

	Mat maze_solved_line = maze_solved_image.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (BTRACE[i][j]) maze_solved_line.data[i * height + j] = 255;
		}
	}

	Mat maze_no_red_line = maze_solved_image.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (BTRACE[i][j]) maze_no_red_line.data[i * height + j] = 0;
		}
	}

	Mat zeros(height, width, CV_8UC1);

	vector<Mat> channels;
	channels.push_back(maze_no_red_line);
	channels.push_back(maze_no_red_line);
	channels.push_back(maze_solved_line);

	end_t = clock();

	Mat result_img;

	merge(channels, result_img);

	namedWindow("Solved Maze Image");
	imshow("Solved Maze Image", result_img);
	imwrite("maze_solved.png", result_img);

	waitKey(0);

	runtime = (double)(end_t - start_t);
	printf("runtime : %f ms\n", runtime);

	cvtColor(maze_solved_image, result_img, COLOR_GRAY2BGR);

	namedWindow("Solved Maze Line");
	imshow("Solved Maze Line", maze_solved_line);
	imwrite("maze_line.png", maze_solved_line);
	
	waitKey(0);

	delete[] IMGMAT;

	for (int i = 0; i < height; ++i) {
		delete[] BTRACE[i];
	}
	delete[] BTRACE;

	return 0;
}

void lee_algorithm(POINT start, POINT end, int** IMGMAT, int length) {

	POINT* C = new POINT[MAX_VEC_SIZE];
	int C_tail = 0;
	int C_front = 0;

	C[C_tail] = start;
	C_tail += 1;

	IMGMAT[start.x][start.y] = 1;

	int dimA;
	dimA = length * length;

	int BLOCK_SIZE = 256;

	int* d_a;

	size_t memSize = dimA * sizeof(int);
	cudaMalloc((void**)&d_a, memSize);

	POINT* C_gpu;
	cudaMalloc((void**)&C_gpu, sizeof(POINT) * MAX_VEC_SIZE);

	POINT* end_gpu;
	cudaMalloc((void**)&end_gpu, sizeof(POINT));

	int* C_gpu_tail;
	cudaMalloc((void**)&C_gpu_tail, sizeof(int));

	int* IMGMAT_endP_gpu;
	cudaMalloc((void**)&IMGMAT_endP_gpu, sizeof(int));

	int* length_gpu;
	cudaMalloc((void**)&length_gpu, sizeof(int));

	cudaStream_t stream0;
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	cudaMemcpyAsync(C_gpu, C, sizeof(POINT) * MAX_VEC_SIZE, cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_a, IMGMAT[0], memSize, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(end_gpu, &end, sizeof(POINT), cudaMemcpyHostToDevice, stream2);
	cudaMemcpyAsync(C_gpu_tail, &C_tail, sizeof(int), cudaMemcpyHostToDevice, stream3);
	cudaMemcpyAsync(length_gpu, &length, sizeof(int), cudaMemcpyHostToDevice, stream2);

	int IMGMAT_endP = 0;

	while (!IMGMAT_endP) {
		int numBlocks = ceil((Q_TAIL_INDEX) / float(BLOCK_SIZE));

		lee_algorithm_cuda << <numBlocks, BLOCK_SIZE, 0, stream0 >> > (C_gpu, C_gpu_tail, end_gpu, IMGMAT_endP_gpu, d_a, length_gpu);

		cudaMemcpyAsync(&IMGMAT_endP, IMGMAT_endP_gpu, sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}
	cudaMemcpy(IMGMAT[0], d_a, memSize, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(end_gpu);
	cudaFree(C_gpu);
	cudaFree(C_gpu_tail);
	cudaFree(IMGMAT_endP_gpu);
	cudaFree(length_gpu);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
}

__global__ void lee_algorithm_cuda(POINT* C, int* C_tail, POINT* end_gpu, int* IMGMAT_endP_gpu, int* IMGMAT, int* length_gpu) { //length 안넘겨도 되는지?
	POINT current, v;
	int length = length_gpu[0];

	__shared__ int near_row[4];
	__shared__ int near_col[4];
	__shared__ POINT C_block[4 * 256]; //4 * Block_size
	__shared__ int C_block_tail;
	__shared__ int C_tail_shared;

	const int old_C_tail = C_tail[0];
	const int old_C_front = 0;

	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	// tail and front : all same in block

	if (threadIdx.x == 0) {
		near_row[0] = -1; near_row[1] = 0; near_row[2] = 1; near_row[3] = 0;
		near_col[0] = 0; near_col[1] = -1; near_col[2] = 0; near_col[3] = 1;
		C_block_tail = 0;
	}
	__syncthreads();
	if (index == 0) {
		C_tail[0] = 0;
	}

	if (index < old_C_tail) {
		current = C[index];
		for (int i = 0; i < 4; i++) {
			v.x = current.x + near_row[i];
			v.y = current.y + near_col[i];
			if (0 <= v.x && v.x < length && 0 <= v.y && v.y < length) {
				const int flag = atomicCAS(&(IMGMAT[v.x * length + v.y]), 0, 0);
				if (flag == 0) {
					IMGMAT[v.x * length + v.y] = IMGMAT[current.x * length + current.y] + 1;
					const int tail_temp = atomicAdd(&C_block_tail, 1);
					C_block[tail_temp] = v;
				}
			}
		}
	}
	__syncthreads();

	if (index == 0) {
		IMGMAT_endP_gpu[0] = IMGMAT[end_gpu[0].x * length + end_gpu[0].y];
	}
	if (threadIdx.x == 0) {
		C_tail_shared = atomicAdd(C_tail, C_block_tail);
	}
	__syncthreads();
	for (int i = threadIdx.x; i < C_block_tail; i += blockDim.x) {
		C[C_tail_shared + i] = C_block[i];
	}
	__syncthreads();
}

void backtrace(POINT start, POINT end, int** IMGMAT, int** BTRACE, int length) {
	POINT current;

	int** BTRACE_min = new int* [length];
	BTRACE_min[0] = new int[length * length];
	for (int i = 1; i < length; i++) {
		BTRACE_min[i] = BTRACE_min[i - 1] + length;
	}
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < length; j++) {
			BTRACE_min[i][j] = 0;
		}
	}

	int dimA;
	dimA = length * length;
	int BLOCK_SIZE = 256;
	int numBlocks = ceil(dimA / float(BLOCK_SIZE));

	cudaStream_t stream1;
	cudaStream_t stream2;

	int* d_a;	//for BTRACE_min
	size_t memSize = dimA * sizeof(int);
	cudaMalloc((void**)&d_a, memSize);

	int* IMGMAT_constant;	//for BTRACE_min
	cudaMalloc((void**)&IMGMAT_constant, memSize);

	int* length_gpu;
	cudaMalloc((void**)&length_gpu, sizeof(int));

	cudaMemcpy(IMGMAT_constant, IMGMAT[0], memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(length_gpu, &length, sizeof(int), cudaMemcpyHostToDevice);

	backtrace_cuda << <numBlocks, BLOCK_SIZE >> > (IMGMAT_constant, d_a, length_gpu);

	cudaMemcpy(BTRACE_min[0], d_a, memSize, cudaMemcpyDeviceToHost);

	BTRACE[end.x][end.y] = 255;
	current = end;

	// 1 2 3
	// 8   4
	// 7 6 5
	while (BTRACE[start.x][start.y] == 0) {
		switch (BTRACE_min[current.x][current.y]) {
		case 0:
			break;
		case 1:
			current.y -= 1; current.x -= 1;
			break;
		case 2:
			current.x -= 1;
			break;
		case 3:
			current.y += 1; current.x -= 1;
			break;
		case 4:
			current.y += 1;
			break;
		case 5:
			current.y += 1; current.x += 1;
			break;
		case 6:
			current.x += 1;
			break;
		case 7:
			current.y -= 1; current.x += 1;
			break;
		case 8:
			current.y -= 1;
			break;
		default:
			break;
		}
		BTRACE[current.x][current.y] = 255;
	}

	cudaFree(d_a);
	cudaFree(IMGMAT_constant);
	cudaFree(length_gpu);

	delete[] BTRACE_min;
}

__global__ void backtrace_cuda(int* IMGMAT_constant, int* BTRACE_min, int* length_gpu) {
	POINT current, v, min;
	__shared__ int near_row[8];
	__shared__ int near_col[8];
	int length = length_gpu[0];

	const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x == 0) {
		near_row[0] = -1; near_col[0] = -1;
		near_row[1] = -1; near_col[1] = 0;
		near_row[2] = -1; near_col[2] = 1;
		near_row[3] = 0; near_col[3] = 1;
		near_row[4] = 1; near_col[4] = 1;
		near_row[5] = 1; near_col[5] = 0;
		near_row[6] = 1; near_col[6] = -1;
		near_row[7] = 0; near_col[7] = -1;
	}
	__syncthreads();

	current.x = index / length;
	current.y = index % length;
	int min_i = -1;
	min = current;

	for (int i = 0; i < 8; i++) {
		v.x = current.x + near_row[i];
		v.y = current.y + near_col[i];
		if (0 <= v.x && v.x < length && 0 <= v.y && v.y < length) {
			if ((IMGMAT_constant[v.x * length + v.y] > 0) && (IMGMAT_constant[v.x * length + v.y] < IMGMAT_constant[min.x * length + min.y])) {
				min = v;
				min_i = i;
			}
		}
	}
	if (min_i != -1) BTRACE_min[current.x * length + current.y] = min_i + 1;
}