#include <bdesn/network.hpp>

namespace BDESN{
void Network::SetInputs( const cv::Mat & inputs )
    {
        if ( inputs.size() != mIn.rows )
            throw std::invalid_argument( "Wrong size of the input vector" );
        //these are all 8 bit so we probably should have promoted to 32 bit before addition but operations on  8 bit mat are faster so done at the end
      	in_m +=  in_m_bias + inputs ;
	in_m *= in_m_scaling;                        
	in_m.convertTo(in_m,CV_32FC2);
    }
}
