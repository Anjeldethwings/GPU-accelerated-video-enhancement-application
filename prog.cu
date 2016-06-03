 #include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int in_final[352][640];
int in_1[352*640];
/*
__global__ void equalize(int In_1[352 * 640], int X, int Y, int rows, int cols){
	int i, j;

	i = blockIdx.y*blockDim.y + threadIdx.y;
	j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < rows && j < cols){
		if (In_1[i*cols + j] == X) Y++;	
	}
}
*/
__global__ void cdf(int T[256], int temp[256], int muls){
	int i = threadIdx.x;
	if (i < 256){
		T[i] = (int)((temp[i]) * 255 / (muls));
	}
}

__global__ void Intensity(unsigned char *Csrc,int In_1[352*640],int ROWS,int COLS){
int rows,cols;
	rows = blockIdx.y*blockDim.y + threadIdx.y;
	cols = blockIdx.x*blockDim.x + threadIdx.x;

	if(rows<ROWS && cols<COLS){

		int p = (int) Csrc[3*(rows*COLS+cols)]; //b
		int q = (int) Csrc[3*(rows*COLS+cols)+1]; //g
		int r = (int) Csrc[3*(rows*COLS+cols)+2]; //r
   
		int a = (int)((p + q + r )/3);
		In_1[rows*COLS + cols] = a ;
		}
}

int main( int argc, char** argv ){

	int n = 0;
	VideoCapture cap("1.mp4"); 

	float b,r,g,in,s,h,h_1;
	float c = (float)3.14159/180 ; //  (pi/180)
	int i, j,a;

//-------------------------------------------------------
	Size S = Size( 640, 352);  

	VideoWriter outVideo;
	outVideo.open(".\\outVideo.avi", 0, cap.get(CV_CAP_PROP_FPS), S, 1); 
//-------------------------------------------------------
	cudaEvent_t start,stop;
	float elapsedtime;

	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

int num = 0 ;
    while(1)
    {
        Mat src;
        bool bsuccess = cap.read(src);

        if(!bsuccess){
        	cout<<num<<endl;
        	break ;
        }

		int rows = src.rows;
		int cols = src.cols;

		Mat hsi(rows, cols, src.type() );// HSI
		Mat final(rows, cols, src.type() );

		namedWindow("Orginal",CV_WINDOW_NORMAL);
		//imshow("Orginal", src);

		  //-------------Intensity--------------------------------------------------
			int *cudaIn_1;
			unsigned char *cudaSrc;

			//allocate memory
			cudaMalloc((void**)&cudaSrc, sizeof(int)*cols*rows*3);
			cudaMalloc((void**)&cudaIn_1, sizeof(int)* (352 * 640));


			//copy inputs	
				cudaMemcpy(cudaSrc, src.data, sizeof(int)*rows*cols*3, cudaMemcpyHostToDevice);


				dim3 numBlocks(ceil(cols / (float)16), ceil(rows / (float)16));
				dim3 threadperblock(16, 16);
				Intensity <<< numBlocks, threadperblock >>> (cudaSrc,cudaIn_1, rows, cols);

				cudaMemcpy(&in_1, cudaIn_1, sizeof(int)* (352*640), cudaMemcpyDeviceToHost);

				cudaFree(cudaIn_1);
				cudaFree(cudaSrc);

				/*

			for(i = 0 ;i < rows;i++){
					  for(j = 0 ; j < cols;j++){
						  int p = src.at<Vec3b>(i,j)[0]; //b
						  int q = src.at<Vec3b>(i,j)[1]; //g
						  int r = src.at<Vec3b>(i,j)[2]; //r
						  a = (int)((p + q + r )/3);
						  in_1[i*cols + j] = a ;
					  }
		  }
		  */
		  //---------------equalize-----------------------------------------
			int hist[256];

			int x ,y ;
			y = 0 ;
			for(x = 0 ;x<256;x++){
			/*
				int *cudaIn_1;

				//allocate memory
				cudaMalloc((void**)&cudaIn_1, sizeof(int)* (352 * 640));

				//copy inputs	
				cudaMemcpy(cudaIn_1, in_1, sizeof(int)*(352 * 640), cudaMemcpyHostToDevice);

				dim3 numBlocks(ceil(cols / (float)16), ceil(rows / (float)16));
				dim3 threadperblock(16, 16);
				equalize <<< numBlocks, threadperblock >>> (cudaIn_1, x, y, rows, cols);

				//free
				cudaFree(cudaIn_1);

				*/
				for (i = 0 ; i < src.rows;i++ ){
					for(j = 0 ;j < src.cols ; j++){
						if(  in_1[i*cols + j] == x ) y++ ;						
					}
				}
				hist[x] = y;
				y = 0 ;
			}

			//-------------Sorting-----------------------
			/*int temp_1 ;
			for(j=0;j<256;j++){
				for(i=0;i<256;i++){
					if(hist[i] > hist[i+1]){
						temp_1=hist[i];
						hist[i]=hist[i+1];
						hist[i+1]=temp_1;
					}
				}
			}*/
			//--------CDF----------------------------------
			int temp[256] ;
			temp[0] = hist[0] ;// cdf
				for (i=1 ; i<256;i++){
					temp[i] = temp [i-1] + hist[i] ;
				}
			//------------------------------------------
			int T[256];
			int muls = (src.rows*src.cols);
			int *cudaT;
			int *cudaTemp;

			cudaMalloc((void**)&cudaT, sizeof(int)* 256);
			cudaMalloc((void**)&cudaTemp, sizeof(int)* 256);

			cudaMemcpy(cudaTemp, temp, sizeof(int)* 256, cudaMemcpyHostToDevice);
			
			cdf <<< 1, 256 >>> (cudaT,cudaTemp, muls);

			cudaMemcpy(&T, cudaT, sizeof(int)* 256, cudaMemcpyDeviceToHost);

			cudaFree(cudaT);
			cudaFree(cudaTemp);

			/* 
			for(i=0;i<256;i++){

				T[i] = (int)( (temp[i]) *255/(src.rows*src.cols)) ;	

			}
			*/

			x = 0;
			for(i = 0 ;i < hsi.rows;i++){ 
				for (j = 0 ; j < hsi.cols;j++ ){
					for(x=0 ; x < 256 ; x++){
						if(  in_1[i*cols + j] == x )  in_final[i][j] = T[x] ;
					}
				}
			}
		  //--------RGB 2 HSI-------------------------------------------------------
		  /*
		  //cuda implementation
		  */

				  for(i = 0 ;i < rows;i++){
					  for(j = 0 ; j < cols;j++){	

						  b =  src.at<Vec3b>(i,j)[0];
						  g =  src.at<Vec3b>(i,j)[1];
						  r =  src.at<Vec3b>(i,j)[2];

						       //---------intensity-----------------------------------------------------
								  in = (b+g+r)/3 ;
							   //--------saturation------------------------------------------------------
								  float min_val = 0;
								  min_val = min(r, min(b,g));
								  if(in > 0)
										 s = 1 - (min_val/in);
								  else if(in == 0)
										  s = 0;
								//---------Hue-----------------------------------------------------------
								  h_1 =  (float)((r - 0.5* g - 0.5* b)) /sqrt(r*r + g*g + b*b - r*g - r*b - g*b );
								  h = acos(h_1);
								  if(b <= g){
									  h = h;
								  }
								  else{
									  h = (22/7)*2 - h;
								  }
								  //----------------------------------------------------------------------
								   hsi.at<Vec3b>(i,j)[2] = in ;
								   hsi.at<Vec3b>(i,j)[1] = s ;
								   hsi.at<Vec3b>(i,j)[0] = h ;
								//---------------------------------------------------------------------
								   in = (float) in_final[i][j];
								//------------------------------------------------------------------------
							   int r_1,g_1,b_1;

								if (h == 0){
								   r_1 = (int) (in + (2 * in * s));
								   g_1 = (int) (in - (in * s));
								   b_1 = (int) (in - (in * s));
								  }

								else if ((0 < h) && (h < 120*c)) {
								   r_1 = (int) (in + (in * s) * cos(h) / cos(60*c-h));
								   g_1 = (int) (in + (in * s) * (1 - cos(h) / cos(60*c-h)));
								   b_1 = (int) (in - (in * s));
								  }

								else if ( h == 120*c ){
								   r_1 = (int) (in - (in * s));
								   g_1 = (int) (in + (2 * in * s));
								   b_1 = (int) (in - (in * s));
								  }

								else if ((120*c < h) && (h < 240*c)) {
								   r_1 = (int) (in - (in * s));
								   g_1 = (int) (in + (in * s) * cos(h-120*c) / cos(180*c-h));
								   b_1 = (int) (in + (in * s) * (1 - cos(h-120*c) / cos(180*c-h)));
								  }

								else if (h == 240*c) {
								   r_1 = (int) (in - (in * s));
								   g_1 = (int) (in - (in * s));
								   b_1 = (int) (in + (2 * in * s));
								  }

								else if ((240*c < h) && (h < 360*c)) {
								   r_1 = (int) (in + (in * s) * (1 - cos(h-240*c) / cos(300*c-h)));
								   g_1 = (int) (in - (in * s));
								   b_1 = (int) (in + (in * s) * cos(h-240*c) / cos(300*c-h));
								  }

								if( g_1>255 ) g_1 = 255;
								if( b_1>255 ) b_1 = 255;
								if( r_1>255 ) r_1 = 255;
								if( g_1 < 0 ) g_1 = 0;
								if( b_1 < 0 ) b_1 = 0;
								if( r_1 < 0 ) r_1 = 0;

								final.at<Vec3b>(i,j)[0] = b_1;
								final.at<Vec3b>(i,j)[1] = g_1;
		    	  				final.at<Vec3b>(i,j)[2] = r_1;
					  //------------------------------------------------------------------------
					  }
				  }
				  outVideo << final; 

				  		num++;
				  		//printf("%d\n",num );
				  		//if(num == 1820) break;

				  // Mat imgH = final + Scalar(50, 50, 50);

				  namedWindow("Final",CV_WINDOW_NORMAL);
				  imshow("Final", final);
				  /*
				  namedWindow("HSI_Convertion",CV_WINDOW_NORMAL);
				  imshow("HSI_Convertion", hsi);
				  */
				  
//-------------------------------------------------------------------------------------------------------
			if (waitKey(10) == 27){
					cout << "esc key is pressed by user" << endl;	
					break;
			}
	}

	//checkCuda(cudaFree(matA_cuda));
	//checkCuda(cudaFree(matB_cuda));
	//checkCuda(cudaFree(ans_cuda));

	cudaEventElapsedTime(&elapsedtime,start,stop);
	fprintf(stderr,"Time spent for operation is %.10f seconds\n",elapsedtime/(float)1000);

  return 0;
}
