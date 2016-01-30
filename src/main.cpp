#include <iostream>
#include "detector.h"

using namespace std;

int main() {
    Detector detector("Traffic in China.mp4");
    detector.detect();
    return 0;
}

