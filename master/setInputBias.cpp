#include <bdesn/network.hpp>

namespace BDESN{
  void Network::SetInputBias( const cv::Mat & bias )
    {
        if ( bias.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );
        in_m_bias = bias ;
    }
}
