#ifndef SEAMFUNCTIONS_H
#define SEAMFUNCTIONS_H

#include "QtOpencvCore.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace seam {
    void sobel(const cv::Mat& myImage, cv::Mat& Result);
} // namespace

#endif // SEAMFUNCTIONS_H
