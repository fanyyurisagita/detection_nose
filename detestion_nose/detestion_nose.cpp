#include "opencv/cv.h"
#include "opencv/highgui.h"

//#include <stdio.h>
#include <iostream>
#include <ctime> // library untuk waktu

// *** Change this to your install location! ***
// *********************************************

//#define OPENCV_ROOT  "C:/opencv"
#define OPENCV_ROOT  "C:/opencv"

// *********************************************

using namespace std;
using namespace cv;

const char * DISPLAY_WINDOW = "Deteksi Hidung";

IplImage *frame;
CvRect* r;
CvRect rect;

double elapsed;
int last_time;
int num_frames;
clock_t start;	// variabel waktu



CvHaarClassifierCascade * pCascade = 0;		// the idung detector
CvMemStorage * pStorage = 0;				// memory for detector to use
CvSeq * idung;								// memory-access interidung	

void overlayImages(CvRect * point, IplImage * frame);
void detect_idung(IplImage * frame, CvSeq * idung);

int main(int argc, char** argv)
{
	cout << "test" << endl;
	CvCapture *capture = cvCaptureFromCAM(0);

	pStorage = cvCreateMemStorage(0);
	pCascade = (CvHaarClassifierCascade *)cvLoad
	   ((OPENCV_ROOT"/data/haarcascades/haarcascade_mcs_nose.xml"),
	   0, 0, 0 );

	// validate that everything initialized properly
	if( !capture || !pStorage || !pCascade )
	{
		printf("Initialization failed: %s\n",
			(!capture)?  "can't load image file" :
			(!pCascade)? "can't load haar-cascade -- "
				         "make sure path is correct" :
			"unable to allocate memory for data storage", argv[1]);
 		exit(-1);
	}

	// create a window to display detected idung
	cvNamedWindow(DISPLAY_WINDOW, CV_WINDOW_AUTOSIZE);

	// Show the image captured from the camera in the window and repeat
	while(1) {
		// mengambil waktu saat algoritma utama mulai dijalankan
		start = clock();

		//cout<<start<<endl;

		// mengambil setiap frame dari video capture
		frame = cvQueryFrame( capture );

		// detect idung in image
		idung = cvHaarDetectObjects
			(frame, pCascade, pStorage,
			1.2,						// increase search scale by 10% each pass
			3,							// merge groups of three detections
			CV_HAAR_DO_CANNY_PRUNING,	// skip regions unlikely to contain a idung
			cvSize(50,50));			// smallest size idung to detect = 50x50

		detect_idung(frame, idung);

		cvShowImage(DISPLAY_WINDOW, frame);
		
		cout << "waktu Komputasi: " << (clock() - start) / (double) CLOCK_PER_SECOND << "\t ms per second" <<endl;
		IplImage * hist = cvCreateImage(cvGetSize(frame),8,1);
		cvCvtColor( frame, hist, CV_BGR2GRAY );
		cvEqualizeHist(hist, hist);
		cvShowImage("hist",hist);


		if ( (cvWaitKey(10) & 255) == 27 ) break;
	}

	// clean up and release resources
	cvReleaseCapture(&capture);
	if(pCascade) cvReleaseHaarClassifierCascade(&pCascade);
	if(pStorage) cvReleaseMemStorage(&pStorage);
	
	cvDestroyWindow("show thumb");
	cvDestroyWindow(DISPLAY_WINDOW);
	cvDestroyWindow("hist");
	return 0;
}


void detect_idung(IplImage * frame, CvSeq * idung)
{
	int i;
	IplImage * src = cvLoadImage("C:/testimage/test.jpg");

	// draw a rectangular outline around each detection
	for(i=0;i<(idung? idung->total:0); i++ )
	{
		int total = idung->total;
		r = (CvRect*)cvGetSeqElem(idung, i);
		
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->height, r->y + r->height };
		cvRectangle(frame, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0);
		
		CvRect recta = cvRect(r->x, r->y, 120, 120);	
		cvSetImageROI(frame, recta );

		IplImage * gray =  cvCreateImage(cvGetSize(frame), frame->depth, 1);
		cvCvtColor(frame,gray,CV_BGR2GRAY);
		cvSaveImage("D:/hidung.jpg", gray);

		cvShowImage("show thumb", gray);

		cvResetImageROI(frame);

		//cvReleaseImage(&gray);
	}
	
	// Mengitung Framerate
	num_frames++;
	elapsed = clock() - last_time;
	int fps = 0;
	fps = floor(num_frames / (float)(1 + (float)elapsed / (float)CLOCKS_PER_SEC));
	num_frames = 0;
	last_time = clock() + 1 * CLOCKS_PER_SEC;

	CvFont font;

	
	//// menampilkan jumlah fps
	char fpsnum[16];
	char myNum2 = fps;
	itoa (myNum2,fpsnum,10);
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 0.5, 1.0, 0, 1, CV_AA);
	cvPutText(frame,"FPS",cvPoint(550,50), &font, cvScalar(0, 255, 0, 0));
	cvPutText(frame,fpsnum,cvPoint(500,50), &font, cvScalar(0, 255, 0, 0));

}
}
