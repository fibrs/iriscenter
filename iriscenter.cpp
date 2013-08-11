#include <cv.h>
 #include <highgui.h> 
 #include <stdio.h>  
#include <sstream>

using namespace cv;
using namespace std;

void print_help()
{
  cout << "USAGE:   iriscenter (camera #) (camera resolution)" << endl << endl << "camera #: Selects the camera to use as input for iris detection." << endl << "   - '0' selects the first camera (typically the integrated webcam)" << endl << "   - defaults to the first available camera if omitted, however the camera # must be specified to select the non-default resolution for the camera." << endl << endl << "camera resolution: selects a resolution to requestion from the camera. Resolutions not supported by the camera may cause an error." << endl << "  - '1': 320x240" << endl << "  - '2': 480x320" << endl << "  - '3': 720x480" << endl << "  - '4': 1280x720" << endl << "  - '5': 1920x1080" << endl << "  - if omitted this defaults to the camera's default resolution."  << endl << endl;

  exit(1);
}

//This function draws a line perpendicular to the points (x1,y1) and (x2,y2) onto the Mat for the 'algorithm' below. 
Mat perpLine(Mat lines_data, int x1, int y1, int x2, int y2)
{
  //cout << "Debug: function 'perpLine' called." << endl; 
  
  float m, m3;

  m = (float)(y1 - y2) / (float)(x1 - x2);

  //cout << "Debug: m = " << m << endl;
  m3 = -1.0 / m;

  //cout << "Debug: Slope = " << y1 - y2 << " / " << x1 - x2 << " = " << m3 << "." << endl;
  int x3 = (x1 + x2) / 2;
  int y3 = (y1 + y3) / 2;
  // The line is defined by (y - y3) = m3(x - x3)
  
  //now find the intercepts where the line crosses the boundary of the picture; x=0, x=99, y=0, and y=99;
  int y_x0 = -1 * m3 * x3 + y3;
  int y_x99 = m3 * (99 - x3) + y3;

  int x_y0, x_y99;
  if( m3 != 0 )
  {
    x_y0 = x3 - (y3 / m3);
    x_y99 = ((99 - y3) / m3) + x3;
  }
  else
  {
    x_y0 = x3;
    x_y99 = x3;  
  }

  //two of these intercepts will be outside the bounds of the picture, the other two will be on the boundary. These six cases should determine the two points at which the line leaves the picture
  int xA, xB, yA, yB;
  if(0 <= y_x0 < 100 && 0 <= y_x99 < 100)
  {
    yA = y_x0;
    xA = 0;
    yB = y_x99;
    xB = 99;
  }
  else if(0 <= x_y0 < 100 && 0 <= x_y99 < 100)
  {
    xA = x_y0;
    yA = 0;
    xB = x_y99;
    yB = 99;
  }
  else if(0 <= y_x0 < 100 && 0 <= x_y0 < 100)
  {
    yA = y_x0;
    xA = 0;
    xB = x_y0;
    yB = 0;
  }
  else if(0 <= y_x99 < 100 && 0 <= x_y99 < 100)
  {
    yA = y_x99;
    xA = 99;
    xB = x_y99;
    yB = 99;
  }
  else if(0 <= y_x0 < 100 && 0 <= x_y99 < 100)
  {
    yA = y_x0;
    xA = 0;
    xB = x_y99;

    yB = 99;
  }
  else if(0 <= y_x99 < 100 && 0 <= x_y0 < 100)
  {
    yA = y_x99;
    xA = 99;
    xB = x_y0;
    yB = 0;
  }
  else
  {
    //if none of these match, something is wrong.
    cout << "Error in function perpLine. Exiting program." << endl;
    exit(1);
  }

  //finally we have the points at the boundary of the picture (xA,yA and xB,yB). Now to draw a faint line between these points.

  LineIterator line(lines_data , Point(xA, yA), Point(xB, yB), 4);

  for(int a = 0; a < line.count; a++, ++line)
  {
    lines_data.at<unsigned short>(line.pos()) = lines_data.at<unsigned short>(line.pos()) + 512;
  }

  return (lines_data);
}
/*
Alright, here's the new algorithm.
Step 1: find an eye in the frame, display it and feed it to...
Step 2: Perform canny edge detection on the eye image, display it and feed it too...
Step 3: (here's the new stuff) For each edge pixel, find all the edge pixels at a specific distance. Draw a faint line perpendicular to and half way between each pixel pair. The lines should accumulate similar to the circles in the HoughCircle function. Display the image with the lines.
Step 4: The point with the most line intersections, determined by intensity, should be the center of the most dominant circle. Mark this point with a small cross on each of the displayed images.
Step Hough: Run HoughCircles too so the results can be compared.
*/

void test_algorithm(Mat original)
{
  Mat gray, eye, eye_resized, eye_gray, eye_canny;
  double scaleFactor = 1.1;
  int minNeighbors=5;

  //  --Begin Step 1--
  string eyes_cascade_loc = "/home/fibrs/build/opencv/share/OpenCV/haarcascades/haarcascade_eye.xml";


  CascadeClassifier eyes_cascade;

  if( !eyes_cascade.load( eyes_cascade_loc ) )
  { 
    printf("--(!)Error loading eyes-cascade\n"); 
    return;
  }

  cvtColor(original, gray, CV_BGR2GRAY);
     
  vector< Rect_<int> > eyes;

  eyes_cascade.detectMultiScale(gray, eyes, scaleFactor, minNeighbors, 0 | CV_HAAR_SCALE_IMAGE, Size (30, 30));

  //Just show the first eye found for now
  if( eyes.size() > 0 )
  {
    Rect eye_i = eyes[0];
    eye = original(eye_i);
       
    cv::resize(eye, eye_resized, Size(100, 100), 1.0, 1.0, INTER_CUBIC);

    imshow("Eye BGR", eye_resized);
    moveWindow("Eye BGR", 100,10);
  }
  else
  {
    cout << "No eyes were found. Try again." << endl;
    return;
  }
  //  --End Step 1--

  //  --Begin Step 2--
  cvtColor( eye_resized, eye_gray, CV_BGR2GRAY );

  blur( eye_gray, eye_gray, Size(3,3) );
  //needed: some way to calulated the canny thresholds based on the image (contrast?)
  Canny( eye_gray, eye_canny, 15, 45, 3 );

  imshow("Eye Canny", eye_canny);
  moveWindow("Eye Canny", 200,10);
  //  --End Step 2--

  //  --Begin Step 3--
  Mat lines_data = Mat::zeros(eye_canny.rows, eye_canny.cols, CV_16U);

  for(int i = 0; i < eye_canny.rows; i++)
  {
    for(int j = 0; j < eye_canny.cols; j++)
    {
      if(eye_canny.at<char>(i,j) != 0)
      {
        
        //lines_data.at<unsigned short>(i,j) = 256;
        
        // First scan the top and bottom of the 9x9 "box"        
        for(int l = -4; l < 5; l = l + 9) //scans pixels 4 above and then 4 below i,j (two total)
        {
          for(int k = -4; k < 5; k++) // scans pixels from 4 left to 4 right (9 each pass)
          {
            if((i+k) >= 0 && (i+k) <= eye_canny.rows && (j+l) >= 0 && (j+l) <= eye_canny.cols) //make sure we're still inside the picture
            {
              if(eye_canny.at<char>(i+k,j+l) != 0)
              {
                lines_data = perpLine(lines_data, i, j, i+k, j+l);
              }
            }
          }
        }

        // next scan the remaining pixels on the side of the "box"        
        for(int k = -4; k < 5; k = k + 9) //scans pixels 4 left and then 4 right i,j (two total)
        {
          for(int l = -3; l < 4; l++) // scans pixels from 3 above to 3 below (7 each pass)
          {    
            if((i+k) >= 0 && (i+k) <= eye_canny.rows && (j+l) >= 0 && (j+l) <= eye_canny.cols) //make sure we're still inside the picture
            {
              if(eye_canny.at<char>(i+k,j+l) != 0)
              {
                lines_data = perpLine(lines_data, i, j, i+k, j+l);
              }
            }
          }
        }
      }
    }
  }
  //  --End Step 3--

  //  --Begin Step 4--
  Point maxPt, minPt;
  double max, min;
  minMaxLoc(lines_data, &min, &max, &minPt, &maxPt, Mat());
  line(lines_data, Point(maxPt.x - 5, maxPt.y), Point(maxPt.x + 5, maxPt.y), 65000, 1); //65000 is white(ish) for this type of Mat
  line(lines_data, Point(maxPt.x, maxPt.y - 5), Point(maxPt.x, maxPt.y + 5), 65000, 1);
  cout << "Maximum value of " << max << " at " << maxPt.x <<"," << maxPt.y << endl;

 //  --End Step 4--

  // Now that the new algorithm is atleast partially working, we'll perform the Houghcircle function on the same input and compare the results.

  //  --Begin Step 'HoughCircles'--
  vector<Vec3f> circles;

  HoughCircles(eye_canny, circles, CV_HOUGH_GRADIENT, 1, 50, 20, 20, 10, 35);

  cout << "HoughCircles method: " << circles.size() << " circles found at:" << endl;
  for(int c = 0; c < circles.size(); c++)
  {
    cout << "     " << circles[c][0] << "," << circles[c][1] << endl;
    line(eye_resized, Point(circles[c][0] - 5, circles[c][1]), Point(circles[c][0] + 5, circles[c][1]), Scalar(0,255,0), 1);
    line(eye_resized, Point(circles[c][0], circles[c][1] - 5), Point(circles[c][0], circles[c][1] + 5), Scalar(0,255,0), 1);    
  }

  imshow("Hough Center", eye_resized);
  moveWindow("Hough Center", 500,10);

  //  --End Step 'HoughCircles'--  

  cout << "New Method: maximum value of " << max << " at " << maxPt.y <<"," << maxPt.x << endl;
  imshow("TEST", lines_data);
  moveWindow("TEST", 300,10);  

  line(eye_resized, Point(maxPt.y - 5, maxPt.x), Point(maxPt.y + 5, maxPt.x), Scalar(0,0,255), 1);
  line(eye_resized, Point(maxPt.y, maxPt.x - 5), Point(maxPt.y, 
maxPt.x + 5), Scalar(0,0,255), 1);
  imshow("Iris Center", eye_resized);
  moveWindow("Iris Center", 400,10);
  cout << "Press [Enter] to capture another frame." << endl;

  return;
}

 int main(int argc, char** argv) {

   int camera = -1;

   bool set_res = 0;
   int res = 0;
   int frame_width = 320;
   int frame_height = 240;

   if(argv[1])
   {
     if(strcmp(argv[1],"-help")==0 || strcmp(argv[1],"--help")==0 || strcmp(argv[1],"-?")==0)
     {
       print_help();
     }

     camera = atoi(argv[1]);
   }

   if(argv[2])
   {
     res = atoi(argv[2]);
     switch ( res ){
     case 0:
       set_res = 0;
       break;
     case 1:
       frame_width = 320;
       frame_height = 240;
       set_res = 1;
       break;
     case 2:
       frame_width = 480;
       frame_height = 320;
       set_res = 1;
       break;
     case 3:
       frame_width = 720;
       frame_height = 480;
       set_res = 1;
       break;
     case 4:
       frame_width = 1280;
       frame_height = 720;
       set_res = 1;
       break;
     case 5:
       frame_width = 1920;
       frame_height = 1080;
       set_res = 1;
       break;
     default:
       print_help;
       break;
     }
   }

   CvCapture* capture = cvCaptureFromCAM( camera );
   //(resolution is set with the next two lines

   if (set_res = 1)
   {
     cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, frame_width);
     cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, frame_height);
   }
   if ( !capture ) {
     cout << stderr << "ERROR: capture " << camera << " is NULL." << endl;
     getchar();
     return -1;
   }
   // Create a window in which the captured images will be presented 
   cvNamedWindow( "mywindow", CV_WINDOW_AUTOSIZE );
 
   Mat original;

   // Show the image captured from the camera in the window and repeat until [enter] is pressed, then take the most recent image and run the new center finding alorgithm on it

   cout << "Press [Enter] to capture a frame and run the test algorithm on it." << endl;

   for(;;)
   {
     original= cvQueryFrame ( capture);
     char key;

     imshow( "mywindow", original );
     moveWindow("mywindow", 100, 150);


     // Do not release the frame!
     //If ESC key pressed, Key=0x10001B under OpenCV 0.9.7(linux version),
     //remove higher bits using AND operator
     //some testing determined that the enter key is 10
     int keypress = cvWaitKey(10);
     //cout << "keypressed = " << keypress << endl;
     if ((keypress & 255) == 10)
     {
       cout << "Here we go!" << endl;
       test_algorithm(original);
     }
     if ( (keypress & 255) == 27 ) break;
   }

   // Release the capture device and housekeeping
   cvReleaseCapture( &capture );
   destroyAllWindows();

   return 0;
 }
