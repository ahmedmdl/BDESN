#include <bdesn/network.hpp>

namespace BDESN{
  void Network::CaptureOutput(cv::Mat  & output )
    {
        if ( output.size() != mParams.outputCount )
            throw std::invalid_argument(
                "Size of the vector must be equal "
                "actual number of outputs" );

	  output = O_m.clone();
    }
}
