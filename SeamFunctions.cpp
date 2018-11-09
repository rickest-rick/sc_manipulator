#include "SeamFunctions.hpp"

void seam::sobel(const cv::Mat& myImage, cv::Mat& Result)
{
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images
    const int nChannels = myImage.channels();
    Result.create(myImage.size(),myImage.type());

    /* compute horizontal sobel f_x */
    for (int j = 1; j < myImage.rows-1; j++)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* next     = myImage.ptr<uchar>(j + 1);
        uchar* output = Result.ptr<uchar>(j);
        for (int i = nChannels; i < nChannels*(myImage.cols-1); i++)
        {
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
    for(int j = 1; j < myImage.rows-1; j++)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current = myImage.ptr<uchar>(j);
        const uchar* next     = myImage.ptr<uchar>(j + 1);
        uchar* output = ResultY.ptr<uchar>(j);
        for(int i= nChannels; i < nChannels*(myImage.cols-1); ++i)
        {
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
}
