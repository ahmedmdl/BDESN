#include <bdesn/network.hpp>

namespace BDESN{
  void Network::CaptureActivations(cv::Mat & activations )
    {
        if ( activations.size() != mParams.neuronCount )
            throw std::invalid_argument(
                "Size of the vector must be equal "
                "actual number of neurons");
	activations = E_m_activation.clone();
    }
}
