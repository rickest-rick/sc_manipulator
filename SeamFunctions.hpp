#ifndef SEAMFUNCTIONS_H
#define SEAMFUNCTIONS_H

#include <vector>
#include <utility>
#include <iostream>

#include "QtOpencvCore.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace seam {
    /**
     * @brief Computes the sobel operator for the image myImage and saves the
     * in Result.
     * @param myImage
     * @param Result - gradients in x and y directions.
     * @details The function computes the energy function by calculating the gradients
     * in x and y directions and saving their sum in Result.
     */
    void sobel(const cv::Mat& myImage, cv::Mat& Result);

    /**
     * @brief Computes the seam in vertical direction with the lowest sum of energy.
     * @param gradientImage - the energy values of a picture.
     * @param blockedPixels - pixels which are blocked and have to be ignored for the seams.
     * @return a seam in vertical direction
     * @details Computes the seam in vertical direction with the lowest sum of energy.
     * For every pixel, only the three neighboring pixels above are considered. To enable
     * the computation of multiple seams, for every row the seams pixel and the two to
     * the left and right are set to 255.
     */
    std::vector<std::pair<quint32, quint32>> seamVertical(cv::Mat& gradientImage,
                                                          std::vector<std::vector<bool>>& blockedPixels);

    /**
     * @brief Computes the seam in horizontal direction with the lowest sum of energy.
     * @param gradientImage - the energy values of a picture.
     * @param blockedPixels - pixels which are blocked and have to be ignored for the seams.
     * @return a seam in horizontal direction
     * @details Computes the seam in horizontal direction with the lowest sum of energy.
     * For every pixel, only the three neighboring pixels to the left are considered. To enable
     * the computation of multiple seams, for every column the seam's pixel and the two above
     * and below are set to 255.
     */
    std::vector<std::pair<quint32,quint32>> seamHorizontal(cv::Mat& gradientImage,
                                                           std::vector<std::vector<bool>>& blockedPixels);
} // namespace

#endif // SEAMFUNCTIONS_H
