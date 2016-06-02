/*
    This is an example illustrating the use of the extract_fhog_features() routine from
    the dlib C++ Library.

    The extract_fhog_features() routine performs the style of HOG feature extraction
    described in the paper:
        Object Detection with Discriminatively Trained Part Based Models by
        P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan
        IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010
    This means that it takes an input image and outputs Felzenszwalb's
    31 dimensional version of HOG features.  We show its use below.
*/

#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace dlib;

int main(int argc, char** argv) {
    cv::VideoCapture cap(1);
    try {
        cv::Mat mat;
        while (cap.read(mat)) {
            // Here we declare an image object that can store color rgb_pixels.
            cv_image<hsi_pixel> img = cv_image<hsi_pixel>(mat);

            // Now convert the image into a FHOG feature image.  The output, hog, is a 2D array
            // of 31 dimensional vectors.
            array2d<matrix<float,31,1> > hog;
            extract_fhog_features(img, hog);

            auto d = draw_fhog(hog);
            cv::Mat f = toMat(d);
            cv::waitKey(10);
            cv::imshow("main", f);
        }
    } catch (exception& e) {
        cout << "exception thrown: " << e.what() << endl;
    }
}
