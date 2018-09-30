#include <bdesn/network.hpp>

namespace BDESN{

  
  void Network::Step( float step )     //learning function
    {
        if ( step <= 0.0f )
            throw std::invalid_argument(
                "Step size must be positive value" );

        #define TEMP in_m_w * in_m + E_m_w * E_m_activation

        #define CALC_E(val) \
	  E_m_activation = smOneMinusLeakingRate * E_m_activation + tanhFunc(mLeakingRate * val, val.size()) 

        #define CALC_E_WITH_FB(val) \
             if (mParams.hasOutputFeedback) CALC_E(TEMP + val); else  CALC_E(TEMP)


	class Parallel_pixel_opencv : public ParallelLoopBody        
           {
           private:
             uchar *p ;
           public:
             Parallel_pixel_opencv(uchar* ptr ) : p(ptr) {}

           virtual void operator()( const Range &r ) const
             {
               for ( register int i = r.start; i != r.end; ++i)
                {
	          p[i] = (int) std::tanh( p[i] )  ;
                }
             }
         };

        void tanhFunc(uchar* matrix, cv::size nElements)
          {
             uchar* pt = matrix.data ;
             parallel_for_( Range(0,nElements) , Parallel_pixel_opencv(pt)) ;
          }



	if ( mParams.linearOutput )
         {
	  CALC_E_WITH_FB(O_m_fb_w * tanhFunc(O_m, O_m.size()) * (O_m_fb_w_scaling));
	  O_m = tanhFunc(O_m_w * E_m_activation, E_m_activation.size())+ prev_O_m;
            prev_O_m = O_m;
	 }
         else
            {
            CALC_E_WITH_FB(O_m_fb_w * O_m * O_m_fb_w_scaling);
            O_m = tanhFunc( O_m_w * E_m_activation, E_m_activation.size()) + prev_O_m;
            prev_O_m = O_m;
	    }

	
	
        #undef TEMP
        #undef CALC_E
        #undef CALC_E_WITH_FB

        auto isnotfinite =
            [] (float n) -> bool { return !std::isfinite(n); };
        if (O_m.unaryExpr(isnotfinite).any())
            throw OutputIsNotFinite();
    }
}
