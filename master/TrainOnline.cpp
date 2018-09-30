#include <bdesn/network.hpp>

namespace BDESN{

  void Network::TrainOnline( const std::vector< float > & output, bool forceOutput )
    {
        for ( unsigned i = 0; i < mParams.outputCount; ++ i )
        {
            Eigen::VectorXf w = mWOut.row( i ).transpose();
            if ( mParams.linearOutput )
                mAdaptiveFilter.Train( w, mOut( i ), output[i], mX );
            else
                mAdaptiveFilter.Train( w, std::atanh( mOut( i ) ),
                    std::atanh( output[i] ), mX );
            mWOut.row( i ) = w.transpose();
        }

        if ( forceOutput )
            mOut = Eigen::Map< Eigen::VectorXf >(
                const_cast< float * >( output.data() ),
                mParams.outputCount );
    }
}
