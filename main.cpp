/**
 * Author : Yeshasvi Tirupachuri
 * Data : 05 Nov 2015
 * Lab : PSPC, Unige, Italy
 * The deafault camera resolution is 640X480
 * @INPROCEEDINGS{991219,
author={Mason, M. and Duric, Z.},
booktitle={Applied Imagery Pattern Recognition Workshop, AIPR 2001 30th},
title={Using histograms to detect and track objects in color video},
year={2001},
pages={154-159},
keywords={computer vision;image colour analysis;image sequences;object detection;color video;histograms;human activity;objects detection;objects tracking;regions of interest;robustness;similarity values;video sequences;Cameras;Color;Face detection;Grid computing;Histograms;Humans;Image edge detection;Layout;Object detection;Video sequences},
doi={10.1109/AIPR.2001.991219},
month={Oct},}
 *
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/assign/std/vector.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//Global Parameters
int kernel_size = 5;
bool first; //To capture the background reference image
bool track; //To do the current frame processing
int x_step = 16, y_step = 12; //size of the cell in the overlay grid

VideoCapture cap(0); // open the default camera
Mat ref_bgr, current_bgr;
Mat ref_gray, current_gray;
Mat bg_image;

//double hist_threshold_int = 400;
//double hist_threshold_chi = 50;

vector<Point> p_1,p_2;
vector<Point>::iterator point1_it,point2_it;

void processCurrent();
void compareHistograms();

/// Initialize trackbar values
int inter_slider = 200;
int chi_slider = 5;

/// Establish the number of bins
int histSize = 256;

/// Set the ranges ( for B,G,R) )
float range[] = { 0, 256 } ;
const float* histRange = { range };

bool uniform = true; bool accumulate = false;

struct cells_bgr{

  cv::Mat cell = cv::Mat(16, 12, CV_8UC3, Scalar(0,0,0)); //Rows and Cols

};

struct cells_gray{

  cv::Mat cell = cv::Mat(16, 12, CV_8UC1, Scalar(0,0,0)); //Rows and Cols

};

struct hists_bgr{

  cv::Mat hist = cv::Mat(16, 12, CV_8UC3, Scalar(0,0,0));

};

struct hists_gray{

  cv::Mat hist = cv::Mat(16, 12, CV_8UC1, Scalar(0,0,0));

};

cells_bgr bg_cell_bgr[40][40];
cells_gray bg_cell_gray[40][40]; //Grid cells for background
hists_gray bg_hist[40][40];
hists_bgr bg_b_hist[40][40], bg_g_hist[40][40], bg_r_hist[40][40];

cells_bgr ct_cell_bgr[40][40];
cells_gray ct_cell_gray[40][40]; //Grid cells for current image
hists_gray ct_hist[40][40];
hists_bgr ct_b_hist[40][40], ct_g_hist[40][40], ct_r_hist[40][40];


int main( int argc, char** argv )
{

  if(!cap.isOpened()){
        cout << "Camera Not Open..." << endl;
        return -1;
    }// check if we succeeded


  int Keyhit=-1;//Used to get Enter input from the keyboard for ROI selection

  while(cap.isOpened()){

      //cout << "Camera is Open!" << endl;
      while(Keyhit==-1){

          cap.read(ref_bgr);
          namedWindow( "Camera Feed", CV_WINDOW_AUTOSIZE );
          putText(ref_bgr, "Press enter to capture background image ", cvPoint(20,60),FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1, CV_AA);
          imshow( "Camera Feed", ref_bgr);
          Keyhit=cv::waitKey(1);

          first = 0;
          track = 0;
      }

      //destroyAllWindows();
      //ref_bgr.copyTo(bg_image);
      //putText(bg_image, "This is the background image ", cvPoint(40,60),FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(255,255,255), 1);
      //namedWindow( "Background Image", CV_WINDOW_AUTOSIZE );
      //imshow( "Background Image", bg_image);
      //waitKey(10);

      while(first == 0){ //If this is the first frame, take it as reference frame
          //cout << "Capturing background frame!" << endl;
          cap.read(ref_bgr);
          cvtColor(ref_bgr,ref_gray,CV_BGR2GRAY);
          first = 1; //Captured first frame
          //namedWindow( "Reference Frame", CV_WINDOW_AUTOSIZE );
          //imshow( "Reference Frame", ref_bgr);
          //waitKey(10);

          //Building Background Model
          //For a 40 X 40 Grid to overlay the whole image each cell is like 16X12 pixels size, Good thing is we know cell size 16X12 pixels
          for(int i = 0; i < 40 ; i++ ){ // Grid Rows
              for(int j = 0; j < 40 ; j++ ){ //Grid Cols

                  //Now store the pixels values in each cell from image
                  for(int pixel_x = 0; pixel_x < 16 ; pixel_x++ ){ //Image Rows
                      for(int  pixel_y = 0; pixel_y < 12 ; pixel_y++ ){ //Image Cols

                          //cout << pixel_x+(x_step*(i)) << " , " << pixel_y+(y_step*(j)) << endl;

                          bg_cell_bgr[i][j].cell.at<uchar>(pixel_x,pixel_y) = ref_bgr.at<uchar>(pixel_x+(x_step*(i)),pixel_y+(y_step*(j)));
                          bg_cell_gray[i][j].cell.at<uchar>(pixel_x,pixel_y) = ref_gray.at<uchar>(pixel_x+(x_step*(i)),pixel_y+(y_step*(j)));


                        }

                    }

                  //cout << cell[i][j].cell << endl;


                  //Now each cell is loaded with proper values from the image, we need to compute the cell histogram now
                  /// Separate the image in 3 places ( B, G and R )
                  vector<Mat> cell_bgr_planes;
                  split( bg_cell_bgr[i][j].cell, cell_bgr_planes );

                  bool uniform = true; bool accumulate = false;

                  /// Compute the histograms
                  calcHist( &bg_cell_gray[i][j].cell, 1, 0, Mat(), bg_hist[i][j].hist, 1, &histSize, &histRange, uniform, accumulate );
                  calcHist( &cell_bgr_planes[0], 1, 0, Mat(), bg_b_hist[i][j].hist, 1, &histSize, &histRange, uniform, accumulate );
                  calcHist( &cell_bgr_planes[1], 1, 0, Mat(), bg_g_hist[i][j].hist, 1, &histSize, &histRange, uniform, accumulate );
                  calcHist( &cell_bgr_planes[2], 1, 0, Mat(), bg_r_hist[i][j].hist, 1, &histSize, &histRange, uniform, accumulate );


                  // Draw the histograms for B, G and R
                  //int hist_w = 512; int hist_h = 400;
                  //int bin_w = cvRound( (double) hist_w/histSize );

                  //Mat cell_histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

                  /// Normalize the result to [ 0, histImage.rows ]
                  //normalize(bg_b_hist[i][j].hist, bg_b_hist[i][j].hist, 0, cell_histImage.rows, NORM_MINMAX, -1, Mat() );
                  //normalize(bg_g_hist[i][j].hist, bg_g_hist[i][j].hist, 0, cell_histImage.rows, NORM_MINMAX, -1, Mat() );
                  //normalize(bg_r_hist[i][j].hist, bg_r_hist[i][j].hist, 0, cell_histImage.rows, NORM_MINMAX, -1, Mat() );

                  /// Draw for each channel
                  /*for( int l = 1; l < histSize; l++ )
                  {
                      line( cell_histImage, Point( bin_w*(l-1), hist_h - cvRound(bg_b_hist[i][j].hist.at<float>(l-1)) ) ,
                                         Point( bin_w*(l), hist_h - cvRound(bg_b_hist[i][j].hist.at<float>(l)) ),
                                         Scalar( 255, 0, 0), 2, 8, 0  );
                      line( cell_histImage, Point( bin_w*(l-1), hist_h - cvRound(bg_g_hist[i][j].hist.at<float>(l-1)) ) ,
                                         Point( bin_w*(l), hist_h - cvRound(bg_g_hist[i][j].hist.at<float>(l)) ),
                                         Scalar( 0, 255, 0), 2, 8, 0  );
                      line( cell_histImage, Point( bin_w*(l-1), hist_h - cvRound(bg_r_hist[i][j].hist.at<float>(l-1)) ) ,
                                         Point( bin_w*(l), hist_h - cvRound(bg_r_hist[i][j].hist.at<float>(l)) ),
                                         Scalar( 0, 0, 255), 2, 8, 0  );
                  }*/



                  /// Display
                  //namedWindow("Cell Histograom", CV_WINDOW_AUTOSIZE );
                  //imshow("Cell Histograom", cell_histImage );

                  //waitKey(10);


                }
            }
          track = 1;
      }

      while(track == 1 ){
          processCurrent();//Current frame processing
        }


 }

  return 0;
}

void processCurrent(){//Current frame processing

  //Process the current frame
  //cout << "Capturing next frame!" << endl;
  cap.read(current_bgr);
  cvtColor(current_bgr,current_gray,CV_BGR2GRAY);
  //namedWindow( "cam feed", CV_WINDOW_AUTOSIZE );
  //imshow( "cam feed", current_bgr);
  //waitKey(10);

  for(int i = 0; i < 40 ; i++ ){ // Grid Rows
      for(int j = 0; j < 40 ; j++ ){ //Grid Cols

          //Nowe store the pixels values in each cell from image
          for(int pixel_x = 0; pixel_x < 16 ; pixel_x++ ){ //Image Rows
              for(int  pixel_y = 0; pixel_y < 12 ; pixel_y++ ){ //Image Cols
                  //cout << "Works and also here..." << endl;

                  //cout << pixel_x+(x_step*(i)) << " , " << pixel_y+(y_step*(j)) << endl;

                  ct_cell_bgr[i][j].cell.at<uchar>(pixel_x,pixel_y) = current_bgr.at<uchar>(pixel_x+(x_step*(i)),pixel_y+(y_step*(j)));
                  ct_cell_gray[i][j].cell.at<uchar>(pixel_x,pixel_y) = current_gray.at<uchar>(pixel_x+(x_step*(i)),pixel_y+(y_step*(j)));
                  //cout << "Works and but here..." << endl;
                  //cout << cell[i][j].cell << endl;


                }

            }


          //Calculating cell histograms for current image
          /// Separate the image in 3 places ( B, G and R )
          vector<Mat> cell_bgr_planes;
          split( ct_cell_bgr[i][j].cell, cell_bgr_planes );

          bool uniform = true; bool accumulate = false;

          /// Compute the histograms
          calcHist( &ct_cell_gray[i][j].cell, 1, 0, Mat(), ct_hist[i][j].hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &cell_bgr_planes[0], 1, 0, Mat(), ct_b_hist[i][j].hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &cell_bgr_planes[1], 1, 0, Mat(), ct_g_hist[i][j].hist, 1, &histSize, &histRange, uniform, accumulate );
          calcHist( &cell_bgr_planes[2], 1, 0, Mat(), ct_r_hist[i][j].hist, 1, &histSize, &histRange, uniform, accumulate );


          // Draw the histograms for B, G and R
          //int hist_w = 512; int hist_h = 400;
          //int bin_w = cvRound( (double) hist_w/histSize );

          //Mat ct_cell_histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

          /// Normalize the result to [ 0, histImage.rows ]
          //normalize(ct_b_hist[i][j].hist, ct_b_hist[i][j].hist, 0, ct_cell_histImage.rows, NORM_MINMAX, -1, Mat() );
          //normalize(ct_g_hist[i][j].hist, ct_g_hist[i][j].hist, 0, ct_cell_histImage.rows, NORM_MINMAX, -1, Mat() );
          //normalize(ct_r_hist[i][j].hist, ct_r_hist[i][j].hist, 0, ct_cell_histImage.rows, NORM_MINMAX, -1, Mat() );

          /// Draw for each channel
          /*for( int l = 1; l < histSize; l++ )
          {
              line( ct_cell_histImage, Point( bin_w*(l-1), hist_h - cvRound(ct_b_hist[i][j].hist.at<float>(l-1)) ) ,
                                 Point( bin_w*(l), hist_h - cvRound(ct_b_hist[i][j].hist.at<float>(l)) ),
                                 Scalar( 255, 0, 0), 2, 8, 0  );
              line( ct_cell_histImage, Point( bin_w*(l-1), hist_h - cvRound(ct_g_hist[i][j].hist.at<float>(l-1)) ) ,
                                 Point( bin_w*(l), hist_h - cvRound(ct_g_hist[i][j].hist.at<float>(l)) ),
                                 Scalar( 0, 255, 0), 2, 8, 0  );
              line( ct_cell_histImage, Point( bin_w*(l-1), hist_h - cvRound(ct_r_hist[i][j].hist.at<float>(l-1)) ) ,
                                 Point( bin_w*(l), hist_h - cvRound(ct_r_hist[i][j].hist.at<float>(l)) ),
                                 Scalar( 0, 0, 255), 2, 8, 0  );
          }*/



          /// Display
          //namedWindow("Current Image Cell Histograom", CV_WINDOW_AUTOSIZE );
          //imshow("Current Image Cell Histograom", ct_cell_histImage );

          //waitKey(10);

        }
    }



  compareHistograms(); //Calling Histogram Compare function

}

void compareHistograms(){

  namedWindow("Tracking", CV_WINDOW_AUTOSIZE);
  createTrackbar("Intersection", "Tracking", &inter_slider, 1000);
  createTrackbar("Chi_Square", "Tracking", &chi_slider, 1000);

  double hist_threshold_int = getTrackbarPos("Intersection", "Tracking");
  double hist_threshold_chi = getTrackbarPos("Chi_Square", "Tracking");

  double inter_b = 0;
  double inter_g = 0;
  double inter_r = 0;
  double inter = 0;

  double chisq_b = 0;
  double chisq_g = 0;
  double chisq_r = 0;
  double chisq = 0;

  //cout <<  hist_threshold_int << "," << hist_threshold_chi << endl;
  //Comparing Histograms
  for(int i = 0; i < 40 ; i++ ){
      for(int j = 0; j < 40 ; j++ ){

           inter_b = compareHist(bg_b_hist[i][j].hist,ct_b_hist[i][j].hist,CV_COMP_INTERSECT);
           chisq_b = compareHist(bg_b_hist[i][j].hist,ct_b_hist[i][j].hist,CV_COMP_CHISQR);

           inter_g = compareHist(bg_g_hist[i][j].hist,ct_g_hist[i][j].hist,CV_COMP_INTERSECT);
           chisq_g = compareHist(bg_g_hist[i][j].hist,ct_g_hist[i][j].hist,CV_COMP_CHISQR);

           inter_r = compareHist(bg_r_hist[i][j].hist,ct_r_hist[i][j].hist,CV_COMP_INTERSECT);
           chisq_r = compareHist(bg_r_hist[i][j].hist,ct_r_hist[i][j].hist,CV_COMP_CHISQR);

           inter = compareHist(bg_hist[i][j].hist,ct_hist[i][j].hist,CV_COMP_INTERSECT);
           chisq = compareHist(bg_hist[i][j].hist,ct_hist[i][j].hist,CV_COMP_CHISQR);

           //cout << "Cell" << "(" << i << "," << j << ")" << ": " << inter << endl;
          //cout << "Cell" << "(" << i << "," << j << ")" << ": " << inter << "," << chisq << endl;
          //cout << "Cell" << "(" << i << "," << j << ")" << ": " << inter_b << "," << chisq_b << "," << inter_g << "," << chisq_g << "," << inter_r << "," << chisq_r << endl;

          if( inter > hist_threshold_int && chisq > hist_threshold_chi){

              Point p1,p2;
              p1.x = i*16; p1.y = j*12;
              p2.x = i*16+16; p2.y = j*12+12;
              //cout << p1.x << "," << p1.y << "," << p2.x << "," << p2.y << endl;
              p_1.push_back(p1); p_2.push_back(p2);


            }
          //cout << compareHist(bg_b_hist[i][j].hist,ct_b_hist[i][j].hist,CV_COMP_CHISQR) << endl;

        }

    }

  //cout << p_1.size() << "," << p_2.size() << endl;

  point2_it = p_2.begin();
  for(point1_it = p_1.begin(); point1_it != p_1.end(); ++point1_it){

      rectangle(current_bgr,*point1_it,*point2_it,255,1,8,0);
      point2_it++;

    }

  p_1.clear(); p_2.clear();

  //namedWindow("Tracking",CV_WINDOW_AUTOSIZE);
  imshow("Tracking",current_bgr);

  waitKey(10);


}

