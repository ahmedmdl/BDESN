#include <bdesn/network.hpp>

namespace BDESN{
  void Network::SetInputScalings( const cv::Mat & scalings )
    {
        if ( scalings.size() != mParams.inputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );
       in_m_scaling = scalings ;
    }
}
