#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>

#define SPEED 10
#define MIN_DIS 0
#define MAX_DIS 100
#define MIN_DIF 5
#define MAX_DIF 30

using namespace cv;
using namespace std;

enum displayMode {LINE, POINT, ORIGIN, PAUSE};

class Detector {
public:
    Detector(const string &video = "");
    void detect();
    void vehicleNum();
private:
    VideoCapture cam;
    void drawLine(Mat pic1, vector<KeyPoint>keyPoints1, Mat pic2, vector<KeyPoint>keyPoints2, vector<DMatch>gMatches, Mat imgMatches);
    void drawPoints(Mat pic1, vector<KeyPoint>gK1, Mat imgPoint1, Mat pic2, vector<KeyPoint>gK2, Mat imgPoint2, Mat imgMatches);
    void drawNothing(Mat pic, Mat pic2, Mat imgMatches);
    int countVehicle(vector<KeyPoint> &gK1, vector<KeyPoint> &gK2, vector<DMatch> &gMatches);
};

Detector::Detector(const string &video) {
    if (video != "") {
        cam.open(video);
    }
}

void Detector::detect() {
    Mat pic1, pic2;
    Ptr<ORB> detector = ORB::create();
    vector<KeyPoint> keyPoints1, keyPoints2, gK1, gK2;
    Mat discriptor1, discriptor2;
    vector<DMatch> matches, gMatches;
    Ptr<DescriptorMatcher> matcher;
    Mat imgMatches, imgPoint1, imgPoint2;
    bool toggle = true;  // determine which of the pic1 and pic2 is the previous picture
    displayMode mode = LINE;  // switch between display modes
    char key = 'l';
    cam >> pic1;
    while (1) {

        for (int i = 0; i < SPEED; ++i) { // catch frame
            if (toggle) {
                cam >> pic2;
                if (pic2.empty()) {
                    return;
                }

            } else {
                cam >> pic1;
                if (pic1.empty()) {
                    return;
                }
            }
        }
        toggle = !toggle;

        if (mode == ORIGIN) {
            drawNothing(pic1, pic2, imgMatches);
        } else {
            detector->detect(pic1, keyPoints1, Mat());
            detector->compute(pic1, keyPoints1, discriptor1);
            detector->detect(pic2, keyPoints2, Mat());
            detector->compute(pic2, keyPoints2, discriptor2);

            matcher = DescriptorMatcher::create("BruteForce-Hamming");
            matcher->match(discriptor1, discriptor2, matches);

            gMatches.clear();
            gK1.clear();
            gK2.clear();
            float maxSize = 0, averageSize = 0;
            for (int i = 0; i < matches.size(); ++i) {
                //cout << "[" << i << "]" << "KeyPoint coordinate: " << keyPoints1[i].pt.x << " " << keyPoints1[i].pt.y << " | response: " << keyPoints1[i].response << " size: " << keyPoints1[i].size << endl;
                if (matches[i].distance < MAX_DIS && matches[i].distance > MIN_DIS) {  // pre-sift
                    float xDif = keyPoints1[matches[i].queryIdx].pt.x - keyPoints2[matches[i].trainIdx].pt.x, yDif = keyPoints1[matches[i].queryIdx].pt.y - keyPoints2[matches[i].trainIdx].pt.y;
                    xDif = xDif >= 0 ? xDif : (-1) * xDif;
                    yDif = yDif >= 0 ? yDif : (-1) * yDif;

                    if ((xDif > MIN_DIF && xDif < MAX_DIF) && (yDif > MIN_DIF && yDif < MAX_DIF)) {
                        //cout << "xDif: " << xDif << ", yDif: " << yDif << endl;
                        gMatches.push_back(matches[i]);
                        gK1.push_back(keyPoints1[i]);
                        gK2.push_back(keyPoints2[i]);
                    }
                    if (keyPoints1[i].size > maxSize) {
                        maxSize = keyPoints1[i].size;
                    }
                    averageSize += keyPoints1[i].size;
                    //cout << "[" << i << "]" << "KeyPoint coordinate: " << keyPoints1[i].pt.x << " " << keyPoints1[i].pt.y << " | response: " << keyPoints1[i].response << " size: " << keyPoints1[i].size << endl;
                    // cout << "Match[" << i << "] distance: " << matches[i].distance << " pushed" << endl;
                }
            }
            //cout << "Total good points: " << gMatches.size() << " " << gK1.size() << " " << gK2.size() << endl;
            //cout << "Max size: " << maxSize << ", Average size: " << averageSize / matches.size() << endl;
            cout << "Vehicle number: " << countVehicle(keyPoints1, keyPoints2, gMatches) << endl;
            //int num = countVehicle(gK1, gK2, gMatches);
            if (mode == LINE) {
                //drawMatches(pic1, keyPoints1, pic2, keyPoints2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                drawMatches(pic1, keyPoints1, pic2, keyPoints2, gMatches, imgMatches);
            }
            if (mode == POINT) {
                drawPoints(pic1, gK1, imgPoint1, pic2, gK2, imgPoint2, imgMatches);
            }


        }
        imshow("Vehicle detector", imgMatches);
        key = cvWaitKey(1);
        if (key == 'l') {
            mode = LINE;
        }
        else if (key == 'p') {
            mode = POINT;
        }
        else if (key == 'o') {
            mode = ORIGIN;
        }
        else if (key == ' ') {
            mode = PAUSE;
        }
        else if (key == 27) {
            return;
        }
        
        if (mode == PAUSE) {
            while (1) {
                key = cvWaitKey(0);
                if (key == ' ') {
                    key = 'l';
                    mode = LINE;
                    break;
                }
                if (key == 'l') {
                    drawLine(pic1, keyPoints1, pic2, keyPoints2, gMatches, imgMatches);
                }
                else if (key == 'p') {
                    drawPoints(pic1, gK1, imgPoint1, pic2, gK2, imgPoint2, imgMatches);
                }
                else if (key == 'o') {
                    drawNothing(pic1, pic2, imgMatches);
                }
                else if (key == 27) {
                    return;
                }
            }
        }
    }
}

void Detector::vehicleNum() {
    Mat pic1, pic2;
    Ptr<ORB> detector = ORB::create();
    vector<KeyPoint> keyPoints1, keyPoints2, gK1, gK2;
    Mat discriptor1, discriptor2;
    vector<DMatch> matches, gMatches;
    Ptr<DescriptorMatcher> matcher;
    Mat imgMatches, imgPoint1, imgPoint2;
    bool toggle = true;  // determine which of the pic1 and pic2 is the previous picture
    int num[5] = {-1}, it = 0, car = 0;;
    cam >> pic1;
    while (1) {
        for (int i = 0; i < SPEED; ++i) { // catch frame
            if (toggle) {
                cam >> pic2;
                
                if (pic2.empty()) {
                    return;
                }
                imshow("frame", pic2);
            } else {
                cam >> pic1;
                if (pic1.empty()) {
                    return;
                }
                imshow("frame", pic1);
            }
        }
        if (cvWaitKey(100) == 27) {
            return;
        }
        toggle = !toggle;
        detector->detect(pic1, keyPoints1, Mat());
        detector->compute(pic1, keyPoints1, discriptor1);
        detector->detect(pic2, keyPoints2, Mat());
        detector->compute(pic2, keyPoints2, discriptor2);

        matcher = DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(discriptor1, discriptor2, matches);

        gMatches.clear();
        gK1.clear();
        gK2.clear();
        for (int i = 0; i < matches.size(); ++i) {
                //cout << "[" << i << "]" << "KeyPoint coordinate: " << keyPoints1[i].pt.x << " " << keyPoints1[i].pt.y << " | response: " << keyPoints1[i].response << " size: " << keyPoints1[i].size << endl;
            if (matches[i].distance < MAX_DIS && matches[i].distance > MIN_DIS) {  // pre-sift
                float xDif = keyPoints1[matches[i].queryIdx].pt.x - keyPoints2[matches[i].trainIdx].pt.x, yDif = keyPoints1[matches[i].queryIdx].pt.y - keyPoints2[matches[i].trainIdx].pt.y;
                xDif = xDif >= 0 ? xDif : (-1) * xDif;
                yDif = yDif >= 0 ? yDif : (-1) * yDif;

                if ((xDif > MIN_DIF && xDif < MAX_DIF) && (yDif > MIN_DIF && yDif < MAX_DIF)) {
                        //cout << "xDif: " << xDif << ", yDif: " << yDif << endl;
                    gMatches.push_back(matches[i]);
                    gK1.push_back(keyPoints1[i]);
                    gK2.push_back(keyPoints2[i]);
                }
            }
        }

        int rawNum = countVehicle(keyPoints1, keyPoints2, gMatches);
        if (num[it] == -1) {
            num[it] = rawNum;
        } else {
            if (rawNum > car + 5) {
                continue;
            }
            num[it] = rawNum;
        }
        if (it == 4) {
            int count = num[0] + num[1] + num[2] + num[3] + num[4];
            count /= 5;
            cout << "Number: " << count << endl;
            it = -1;
        }
        it++;
    }
}


void Detector::drawLine(Mat pic1, vector<KeyPoint>keyPoints1, Mat pic2, vector<KeyPoint>keyPoints2, vector<DMatch>gMatches, Mat imgMatches) {
    drawMatches(pic1, keyPoints1, pic2, keyPoints2, gMatches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Vehicle detector", imgMatches);
}

void Detector::drawPoints(Mat pic1, vector<KeyPoint>gK1, Mat imgPoint1, Mat pic2, vector<KeyPoint>gK2, Mat imgPoint2, Mat imgMatches) {
    drawKeypoints(pic1, gK1, imgPoint1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(pic2, gK2, imgPoint2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawMatches(imgPoint1, vector<KeyPoint>(), imgPoint2, vector<KeyPoint>(), vector<DMatch>(), imgMatches);
    imshow("Vehicle detector", imgMatches);
}

void Detector::drawNothing(Mat pic1, Mat pic2, Mat imgMatches) {
    drawMatches(pic1, vector<KeyPoint>(), pic2, vector<KeyPoint>(), vector<DMatch>(), imgMatches);
    imshow("Vehicle detector", imgMatches);
}


int Detector::countVehicle(vector<KeyPoint> &gK1, vector<KeyPoint> &gK2, vector<DMatch> &gMatches) {
    vector <vector<DMatch> > vehicle;
    bool *mark = new bool[gMatches.size()];
    for (int i = 0; i < gMatches.size(); ++i) {
        mark[i] = true;
    }

    int number = 0, num = 0;;
    for (int i = 0; i < gMatches.size(); ++i) {
        if (mark[i] == false) {
            continue;
        }
        vehicle.push_back(vector<DMatch>());
        vehicle[number].push_back(gMatches[i]);
        //  cout << vehicle.size() << " " << vehicle[number].size() << endl;
        mark[i] = false;
        for (int j = 1; j < gMatches.size(); ++j) {
            if (mark[j] == false) {
                continue;
            }
            for (int k = 0; k < vehicle[number].size(); ++k) {
                
                float xDif = gK1[vehicle[number][k].queryIdx].pt.x - gK1[gMatches[j].queryIdx].pt.x, yDif = gK1[vehicle[number][k].queryIdx].pt.y - gK1[gMatches[j].queryIdx].pt.y;
                if (xDif > -10 && xDif < 10 && yDif > -10 && yDif < 10) {
                    // cout << number << " " <<  vehicle[number].size() << " " << j << endl;
                    vehicle[number].push_back(gMatches[j]);
                    mark[j] = false;
                    j = 1;
                    break;
                }
            }
        }
       
        if (vehicle[number].size() > 5) {
            num++;
        }
        number++;
        //cout << number << " " << num << endl;
    }
    return num;
}
