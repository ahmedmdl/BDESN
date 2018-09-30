#include <bdesn/network.hpp>

namespace ESN{
  
  void Network::Train(const cv::Mat & inputs,const cv::Mat & outputs )
    {
        if ( inputs.size() == 0 )
            throw std::invalid_argument(
                "Number of samples must be not null" );
        if ( inputs.size() != outputs.size() )
            throw std::invalid_argument(
                "Number of input and output samples must be equal" );

	const unsigned kSampleCount = inputs.size();

        cv::Mat matX(mParams.neuronCount, kSampleCount,CV_32FC2 );
        cv::Mat matY(mParams.outputCount, kSampleCount,CV_32FC2 );

	for ( int i = 0; i < kSampleCount; ++ i )
        {
            SetInputs( inputs[i] );
            Step( 0.1f );
            E_m_activation.copyto( matX.col(i) );
	    outputs[i].copyto( matY.col(i) )
	}

	cv::Mat matXT = matX.t();

        O_m_w = ( matY * matXT * ( matX * matXT ).t() );
    }
}
