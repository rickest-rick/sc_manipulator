#ifndef SEAMFUNCTIONS_H
#define SEAMFUNCTIONS_H

#include <vector>
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
     * 
     * @details The function computes the energy function by calculating the gradients
     * in x and y directions and saving their sum in Result.
     */
    void sobel(const cv::Mat& myImage, cv::Mat& Result);

    /**
     * @brief Computes the seam in vertical direction with the lowest sum of energy.
     * @param gradientImage - the energy values of a picture.
     * @param blockedPixels - pixels which are blocked and have to be ignored for the seams.
     * @return a seam in vertical direction
     * 
     * @details Computes the seam in vertical direction with the lowest sum of energy.
     * For every pixel, only the three neighboring pixels above are considered. To enable
     * the computation of multiple seams, for every row the seams pixel and the two to
     * the left and right are set to 255. The row of each pixel is implicitly stored in the
     * index of the vector.
     */
    std::vector<int> seamVertical(cv::Mat& gradientImage, std::vector<std::vector<bool>>& blockedPixels);

    /**
     * @brief Computes the seam in horizontal direction with the lowest sum of energy.
     * @param gradientImage - the energy values of a picture.
     * @param blockedPixels - pixels which are blocked and have to be ignored for the seams.
     * @return a seam in horizontal direction
     * 
     * @details Computes the seam in horizontal direction with the lowest sum of energy.
     * For every pixel, only the three neighboring pixels to the left are considered. To enable
     * the computation of multiple seams, for every column the seam's pixel and the two above
     * and below are set to 255. The column of each pixel is implicitly stored in the index of
     * the vector.
     */
    std::vector<int> seamHorizontal(cv::Mat& gradientImage, std::vector<std::vector<bool>>& blockedPixels);
    
    /**
     * @brief Downscale an image in vertical direction using the provided seams.
     * @param input
     * @param output - The matrix image where the down-scaled image is saved in.
     * @param verticalSeams The seams in vertical direction which have to be removed -
     * 		  ensure they are sorted
     * 
     * @details 
     */
    void deleteSeamsVertical(const cv::Mat& input, cv::Mat& output, 
                             const std::vector<std::vector<int>>& verticalSeams);
    /**
     * @brief Downscale an image in horizontal direction using the provided seams.
     * @param input
     * @param output - The matrix image where the downscaled image is saved in.
     * @param verticalSeams The seams in horizontal direction which have to be removed -
     * 		  ensure they are sorted
     * 
     */
    void deleteSeamsHorizontal(const cv::Mat& input, cv::Mat& output, 
                             const std::vector<std::vector<int>>& verticalSeams);

    /**
     * @brief Adjust horizontal seams based on vertical seams, which were calculated on the same picture.
     * @param verticalSeams
     * @param horizontalSeams
     * @attention This method has it flaws, if single horizontal and vertical seams have long overlapping parts, for
     * 			  a vertical seam and horizontal seam which both go from the top left to the bottom right.
     */
    void combineVerticalHorizontalSeams(const std::vector<std::vector<int>>& verticalSeams,
                                          std::vector<std::vector<int>>& horizontalSeams);
} // namespace

#endif // SEAMFUNCTIONS_H
