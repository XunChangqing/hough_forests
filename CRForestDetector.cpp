/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "CRForestDetector.h"
#include <vector>
#include "timer.h"


using namespace std;

void CRForestDetector::detectColor(
    IplImage                 *img,
    vector<IplImage*>        &imgDetect,
    std::vector<float> const &ratios)
{
    cdmh::timer t("CRForestDetector::detectColor");

    // extract features
    vector<IplImage*> features;
    CRPatch::extractFeatureChannels(img, features);
    accumulate_votes({img->width, img->height}, features, ratios, imgDetect);
}

std::vector<std::vector<std::vector<LeafNode const *>>>
CRForestDetector::run_regression(
    CvSize                 const &size,
    std::vector<IplImage*> const &features)
{
    cdmh::timer t("CRForestDetector::run_regression");

    std::vector<std::vector<std::vector<LeafNode const *>>>
    results(size.height-height, std::vector<std::vector<LeafNode const *>>(size.width-width));

    // get pointers to feature channels
    int stepImg = 0;
    uchar** ptFCh     = new uchar*[features.size()];
    uchar** ptFCh_row = new uchar*[features.size()];
    for (unsigned int c=0; c<features.size(); ++c)
        cvGetRawData(features[c], (uchar**)&(ptFCh[c]), &stepImg);
    stepImg /= sizeof(ptFCh[0][0]);

    int const xoffset = width/2;
    int const yoffset = height/2;

    for (int y=0, cy=yoffset; y<size.height-height; ++y, ++cy)
    {
        // Get start of row
        for (unsigned int c=0; c<features.size(); ++c)
            ptFCh_row[c] = &ptFCh[c][0];
        
        for (int x=0, cx=xoffset; x<size.width-width; ++x, ++cx)
        {
            // regression for a single patch
            crForest_.regression(results[y][x], ptFCh_row, stepImg);

            // increase pointer - x
            for (size_t c=0; c<features.size(); ++c)
                ++ptFCh_row[c];
        } // end for x

        // increase pointer - y
        for(unsigned int c=0; c<features.size(); ++c)
            ptFCh[c] += stepImg;

    } // end for y 	

    delete[] ptFCh;
    delete[] ptFCh_row;
    return results;
}

void CRForestDetector::accumulate_votes(
    CvSize                 const &size,
    std::vector<IplImage*> const &features,
    std::vector<float>     const &ratios,
    std::vector<IplImage*>       &imgDetect)
{
    accumulate_votes(size, ratios, run_regression(size,features), imgDetect);

}

void CRForestDetector::accumulate_votes(
    CvSize                                                  const &size,
    std::vector<float>                                      const &ratios,
    std::vector<std::vector<std::vector<LeafNode const *>>> const &regression,
    std::vector<IplImage*>                                        &imgDetect)
{
    cdmh::timer t("CRForestDetector::accumulate_votes");

    // reset output image
    for(int c=0; c<(int)imgDetect.size(); ++c)
        cvSetZero(imgDetect[c]);

    // get pointer to output image
    int stepDet = 0;
    float** ptDet = new float*[imgDetect.size()];
    for(unsigned int c=0; c<imgDetect.size(); ++c)
        cvGetRawData( imgDetect[c], (uchar**)&(ptDet[c]), &stepDet);
    stepDet /= sizeof(ptDet[0][0]);

    int const xoffset = width/2;
    int const yoffset = height/2;
    for (int y=0, cy=yoffset; y<size.height-height; ++y, ++cy)
    {
        for (int x=0, cx=xoffset; x<size.width-width; ++x, ++cx)
        {
            // regression for a single patch
            vector<const LeafNode*> const &results = regression[y][x];

            // vote for all trees (leafs) 
            for (auto const &result : results)
            {
                // To speed up the voting, one can vote only for patches 
                // with a probability for foreground > 0.5
                // !!! CH This was commented out in the original code, with no
                //        indication why. It reduces processing from 4m to 20s
                //        on a debug build, so worth having, and produces a
                //        good result. I haven't compared the accuracy fully
                //        yet, though
                if (result->pfg > 0.5)
                {
                    // voting weight for leaf 
                    float w = result->pfg / float( result->vCenter.size() * results.size());

                    // vote for all points stored in the leaf
                    for (auto const &centre : result->vCenter)
                    {
                        for (int c=0; c<(int)imgDetect.size(); ++c)
                        {
                            int const y = cy - centre[0].y;
                            if (y >= 0  &&  y < imgDetect[c]->height)
                            {
                                int const x = int(cx - centre[0].x * ratios[c] + 0.5);
                                if (x >= 0  &&  x < imgDetect[c]->width)
                                    *(ptDet[c]+x+y*stepDet) += w;
                            }
                        }
                    }
                }
            }
        }
    }

    for(int c=0; c<(int)imgDetect.size(); ++c)
        cvSmooth( imgDetect[c], imgDetect[c], CV_GAUSSIAN, 3);
    
    delete[] ptDet;
}

void CRForestDetector::detectPyramid(IplImage *img, vector<vector<IplImage*> >& vImgDetect, std::vector<float> const &ratios) {

    if(img->nChannels==1) {

        std::cerr << "Gray color images are not supported." << std::endl;

    } else { // color

        for(int i=0; i<int(vImgDetect.size()); ++i) {
            IplImage* cLevel = cvCreateImage( cvSize(vImgDetect[i][0]->width,vImgDetect[i][0]->height) , IPL_DEPTH_8U , 3);				
            cvResize( img, cLevel, CV_INTER_LINEAR );	

            // detection
            detectColor(cLevel,vImgDetect[i],ratios);

            cvReleaseImage(&cLevel);
        }

    }

}








