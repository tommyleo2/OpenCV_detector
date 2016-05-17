#ifndef VEHICLEDETECTOR_H
#define VEHICLEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <list>

#define MAX_DIS 100 //distance in DMatch
#define MAX_DIFF 30
#define MIN_DIFF 0   //KeyPoints distance

using namespace std;
using namespace cv;

class VehicleDetector {
public:
    VehicleDetector(bool _start);
    int findVehicle(Mat &frame1, Mat &frame2);

    list< vector<KeyPoint> > KP;
    list<Mat> DP;
private:
    bool start;

    Ptr<ORB> detector;
    Ptr<DescriptorMatcher> matcher;

};

#endif /* VEHICLEDETECTOR_H */
