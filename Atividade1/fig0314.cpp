// This file must be compiled in the following manner: g++ fig0314.cpp -o fig0314.exe `pkg-config --cflags --libs opencv4`
// Including the opencv library used to open, manipulate and write the image
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// This function is used to extract a bitplane 'k'.
// It shifts 1 by k bits e.g (1 << 2) = 100
// Then a bitwise AND operation is performed with the result of the shift and each pixel
// of the Matrix. This results in a new image in which every pixel is an 8 bit value that contains 
// only the value in that selected bitplane with the rest being 0.
Mat extractBitPlane(const Mat &img, int k){
	Mat bitPlane = (img & (1 << k));
	bitPlane = bitPlane * 255 / (1 << k);
	return bitPlane;
}
int main(){

	Mat ORIGINAL_IMAGE = imread("Fig0314(a)(100-dollars).tif", IMREAD_GRAYSCALE);
	
	// We use a for loop that goes from 0 to 7 to extract bitplanes
	// 0 to 7
	for(int k = 0; k < 8; k++){
	Mat plane = extractBitPlane(ORIGINAL_IMAGE, k);
	//The image is then shown and saved in the results folder
	imshow("bitplane_"+to_string(k)+".png", plane);
	waitKey(0);
	destroyAllWindows();
	imwrite("results/100-dollars_bitplane_" + to_string(k) + ".png", plane);
	}



	return 0;
}
