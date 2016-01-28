#include <iostream>
#include "detector.h"

using namespace std;

int main() {
    Detector detector("video.avi");
    detector.detect();
    return 0;
}

