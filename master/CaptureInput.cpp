#include <bdesn/network.hpp>

namespace BDESN{
  void Network::CaptureTransformedInput(cv::Mat & input )
    {
        if ( input.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Size of the vector must be equal to "
                "the number of inputs" );
	input = in_m.clone();
    }
}
