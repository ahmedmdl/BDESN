#include <bdesn/network.hpp>

namespace BDESN{
  void Network::SetFeedbackScalings(
        const cv::Mat & scalings )
    {
        if (!mParams.hasOutputFeedback)
            throw std::logic_error(
                "Trying to set up feedback scaling for a network "
                "which doesn't have an output feedback");
        if ( scalings.size() != mParams.outputCount )
            throw std::invalid_argument(
                "Wrong size of the scalings vector" );
        O_m_fb_w_scaling = scalings;
    }
}
