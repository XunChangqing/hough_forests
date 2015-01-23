/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include "CRForest.h"


class CRForestDetector {
public:
	// Constructor
	CRForestDetector(CRForest const &RF, int w, int h) : crForest_(RF), width(w), height(h)  {}

	// detect multi scale
	void detectPyramid(IplImage *img, std::vector<std::vector<IplImage*> >& imgDetect, std::vector<float> const &ratios);

	// Get/Set functions
	size_t GetNumCenter() const {return crForest_.GetNumCenter();}

    void accumulate_votes(CvSize                 const &size,
                          std::vector<IplImage*> const &features,
                          std::vector<float>     const &ratios,
                          std::vector<IplImage*>       &imgDetect);

    void accumulate_votes(CvSize                                                  const &size,
                          std::vector<IplImage*>                                  const &features,
                          std::vector<float>                                      const &ratios,
                          std::vector<std::vector<std::vector<LeafNode const *>>> const &regression,
                          std::vector<IplImage*>                                        &imgDetect);

    std::vector<std::vector<std::vector<LeafNode const *>>>
    run_regression(CvSize const &size, std::vector<IplImage*> const &features);

private:
    CRForestDetector &operator=(CRForestDetector const &) = delete;
	void detectColor(IplImage *img, std::vector<IplImage*>& imgDetect, std::vector<float> const &ratios);

	CRForest const &crForest_;
	int const width;
	int const height;
};
