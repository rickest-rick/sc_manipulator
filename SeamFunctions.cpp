#include "SeamFunctions.hpp"

void seam::sobel(const cv::Mat& myImage, cv::Mat& Result)
{
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images
    const int nChannels = myImage.channels();
    Result.create(myImage.size(),myImage.type());

    /* compute horizontal sobel f_x */
    for (int j = 1; j < myImage.rows-1; j++) {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* next     = myImage.ptr<uchar>(j + 1);
        uchar* output = Result.ptr<uchar>(j);
        for (int i = nChannels; i < nChannels*(myImage.cols-1); i++) {
            *output++ = cv::saturate_cast<uchar>(next[i-nChannels] + 2 * next[i] + next[i+nChannels]
                    - previous[i-nChannels] - 2 * previous[i] - previous[i+nChannels]);
        }
    }

    /* set edges to neighbouring values */
    Result.row(1).copyTo(Result.row(0));
    Result.row(Result.rows-2).copyTo(Result.row(Result.rows-1));
    Result.col(1).copyTo(Result.col(0));
    Result.col(Result.cols-2).copyTo(Result.col(Result.cols-1));

    /* compute vertical sobel */
    cv::Mat ResultY;
    ResultY.create(myImage.size(),myImage.type());
    for(int j = 1; j < myImage.rows-1; j++) {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current = myImage.ptr<uchar>(j);
        const uchar* next     = myImage.ptr<uchar>(j + 1);
        uchar* output = ResultY.ptr<uchar>(j);
        for(int i= nChannels; i < nChannels*(myImage.cols-1); ++i){
            *output++ = cv::saturate_cast<uchar>(previous[i+nChannels] + 2 * current[i+nChannels]
                    + next[i+nChannels] - previous[i-nChannels] - 2 * current[i-nChannels]
                    - next[i-nChannels]);
        }
    }
    /* set edges to neighbouring values */
    Result.row(1).copyTo(Result.row(0));
    Result.row(Result.rows-2).copyTo(Result.row(Result.rows-1));
    Result.col(1).copyTo(Result.col(0));
    Result.col(Result.cols-2).copyTo(Result.col(Result.cols-1));

    /* add vertical gradients to horizontal gradients on input image */
    cv::add(Result, ResultY, Result);
    ResultY.release();
}

std::vector<int> seam::seamVertical(cv::Mat& gradientImage, std::vector<std::vector<bool>>& blockedPixels)
{
    const int nrows = gradientImage.rows, ncols = gradientImage.cols;
    CV_Assert(gradientImage.depth() == CV_8UC1);  // accept only uchar single channel images
    /* create 2D vector for energy sums with additional border to prevent edge cases */
    std::vector<std::vector<ulong>> energySum(nrows, std::vector<ulong>(ncols + 2, UINT_MAX));
    /* initialize first row */
    for (int i = 0; i < ncols; i++) {
        energySum[0][i+1] = blockedPixels[0][i+1] ? UINT_MAX : gradientImage.at<uchar>(0, i);
    }
    /* Compute energy sum via: E[i,j] = G[i,j] + min{E[i-1,j-1], E[i-1,j], E[i-1,j-1]} */
    for (int i = 1; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            /* mind offset for column because of border */
            ulong energyValue = static_cast<ulong>(gradientImage.at<uchar>(i,j));
            /* pixel can't be used, if it is blocked or if it would cause a crossing of seams */
            if (blockedPixels[i-1][j+1]) {
                if (blockedPixels[i][j])
                    energySum[i][j+1] = energyValue + energySum[i-1][j+2];
                else if (blockedPixels[i][j+2])
                    energySum[i][j+1] = energyValue + energySum[i-1][j];
            } else {
                energySum[i][j+1] = blockedPixels[i][j+1] ? UINT_MAX : energyValue + std::min(energySum[i-1][j],
                        std::min(energySum[i-1][j+1], energySum[i-1][j+2]));
            }
        }
    }
    /* backtrack the seam with the lowest energy sum and set seam to UCHAR_MAX on gradient image */
    std::vector<int> result(nrows);
    std::vector<ulong> lastRow = energySum[nrows-1];
    int col = std::min_element(lastRow.begin(), lastRow.end()) - lastRow.begin(); // start column index
    gradientImage.at<uchar>(nrows-1, col-1) = UCHAR_MAX;
    /* block pixel of seam and the two neighbours to the left and right to prevent crossing */
    blockedPixels[nrows-1][col] = true;
    result[nrows-1] = col - 1;
    if (col == 0) /* @todo error handling if no new seams can be computed */
        std::cout << "col: " << col - 1 << std::endl;
    for (int i = nrows-2; i >= 0; i--) {
        /* find next column index: I[i,j] = argmin{I[i+1,j-1], I[i+1,j], I[i+1, j+1]} and
            delete seam by setting it on high values. */
        std::vector<ulong>* row = &energySum[i];
        col = std::min_element(row->begin() + col - 1, row->begin() + col + 2)
                - row->begin();
        gradientImage.at<uchar>(i,col-1) = UCHAR_MAX;
        /* block pixel of seam and the two neighbours to the left and right to prevent crossing */
        blockedPixels[i][col] = true;
        result[i] = col - 1;
    }
    return result;
}

std::vector<int> seam::seamHorizontal(cv::Mat& gradientImage, std::vector<std::vector<bool>>& blockedPixels)
{
    const int nrows = gradientImage.rows, ncols = gradientImage.cols;
    CV_Assert(gradientImage.depth() == CV_8UC1);  // accept only uchar single channel images
    /* create 2D vector for energy sums with additional border to prevent edge cases */
    std::vector<std::vector<ulong>> energySum(nrows + 2, std::vector<ulong>(ncols, UINT_MAX));
    /* initialize first col */
    for (int i = 0; i < nrows; i++) {
        energySum[i+1][0] = blockedPixels[i+1][0] ? UINT_MAX : gradientImage.at<uchar>(i, 0);
    }
    /* Compute energy sum via: E[i,j] = G[i,j] + min{E[i-1,j-1], E[i,j-1], E[i+1,j-1]} */
    for (int j = 1; j < ncols; j++) {
        for (int i = 0; i < nrows; i++) {
            /* mind offset for column because of border */
            ulong energyValue = static_cast<ulong>(gradientImage.at<uchar>(i,j));
            /* pixel can't be used, if it is blocked or if it would cause a crossing of seams */
            if (blockedPixels[i+1][j-1]) {
                if (blockedPixels[i][j])
                    energySum[i+1][j] = energyValue + energySum[i+2][j-1];
                else if (blockedPixels[i+2][j])
                    energySum[i+1][j] = energyValue + energySum[i][j-1];
            } else {
                energySum[i+1][j] = blockedPixels[i+1][j] ? UINT_MAX : energyValue + std::min(energySum[i][j-1],
                        std::min(energySum[i+1][j-1], energySum[i+2][j-1]));
            }
        }
    }
    /* backtrack the seam with the lowest energy sum and set seam to UCHAR_MAX on gradient image */
    std::vector<int> result(ncols);
    uint row = 0, min = UINT_MAX;
    for (int i = 1; i <= nrows; i++) {
        if (energySum[i][ncols-1] < min) {
            min = energySum[i][ncols-1];
            row = i;
        }
    }
    if (row == 0) /* @todo error handling if no new seams can be computed */
        std::cout << "row: " << row - 1 << std::endl;
    gradientImage.at<uchar>(row-1, ncols-1) = UCHAR_MAX;
    /* block pixel of seam and the two neighbours to the left and right to prevent crossing */
    blockedPixels[row][ncols-1] = true;
    result[ncols-1] = row - 1;
    for (int i = ncols-2; i >= 0; i--) {
        /* find next column index: I[i,j] = argmin{I[i-1,j+1], I[i,j+1], I[i+1, j+1]} and
            delete seam by setting it on high values/blocking pixels. */
        min = UINT_MAX;
        int nextRow = 0;
        for (int j = -1; j <= 1; j++) {
            if (energySum[row+j][i] < min){
                min = energySum[row+j][i];
                nextRow = row + j;
            }
        }
        row = nextRow;
        gradientImage.at<uchar>(row-1, i) = UCHAR_MAX;
        /* block pixel of seam and the two neighbours to the left and right to prevent crossing */
        blockedPixels[row][i] = true;
        result[i] = row - 1;
    }
    return result;
}

void seam::deleteSeamsVertical(const cv::Mat& input, cv::Mat& output, const std::vector<std::vector<int>>& seams)
{
    const int newNumberOfCols = input.cols - seams.size();
    const int nChannels = input.channels();
    output.create(input.rows, newNumberOfCols, input.type());

    /* for every row, copy all values from input, while ignoring pixels from seams */
    for (int i = 0; i < input.rows; i++) {
        int seamOffset = 0; /* number of seams in this row, that were already crossed */
        const uchar* inputRow = input.ptr<uchar>(i);
        uchar* outputRow = output.ptr<uchar>(i);
        for (int j = 0; j < nChannels * newNumberOfCols; j++) {
            if (static_cast<size_t>(seamOffset) < seams.size() && seams[seamOffset][i] * nChannels <= j)
                seamOffset++;
            *outputRow = inputRow[j + seamOffset * nChannels];
            outputRow++;
        }
    }
}

void seam::deleteSeamsHorizontal(const cv::Mat& input, cv::Mat& output, const std::vector<std::vector<int>>& seams)
{
    const int newNumberOfCols = input.cols - seams.size();
    const int nChannels = input.channels();
    int newNumberOfRows = input.rows - seams.size();
    output.create(newNumberOfRows, input.cols, input.type());

    /* for every column, copy all values from input, while ignoring pixels from seams */
    for (int j = 0; j < input.cols; j++) {
        int seamOffset = 0; /* number of seams in this column, that were already crossed */
        for (int i = 0; i < newNumberOfRows; i++) {
            if (static_cast<size_t>(seamOffset) < seams.size() && seams[seamOffset][j] <= i)
                seamOffset++;
            output.at<cv::Vec3b>(i,j) = input.at<cv::Vec3b>(i + seamOffset, j);
        }
    }

}
