  //  mIn => in_m  input matrix 
  //  mWIn => in_m_w   input matrix weight
  //  mWInScaling => in_m_scaling input matrix scaling
  //  mWInBias    => in_m_bias input matrix bias
  //  mW => E_m_w  echo matrix weights
  //  mWFB => O_m_fb_w output matrix feedback weights
  //  mWFBScaling => O_m_fb_w_scaling   output matrix feedback scaling
  //  mX => E_m_activation echo matrix activation
  //  mOut => O_m  output matrix
  //  mWOut => O_m_w output matrix weight
//  Forgetting factor
// initial covariance

#include <bdesn/network.hpp>


namespace BDESN {
  

    std::unique_ptr< Network > CreateNetwork(
        const NetworkParams & params )
    {
        return std::unique_ptr< Network >( new Network( params ) );
    }

    Network::Network( const NetworkParams & params )
      : mParams( params )                                          //basic constructor initializtion
        , in_m( params.inputCount )                                  
        , in_m_w( params.neuronCount,
		  params.inputCount )
        , in_m_scaling( params.inputCount )
        , in_m_bias( params.inputCount )
        , E_m_activation( params.neuronCount )
        , E_m_w( params.neuronCount,
	         params.neuronCount )
        , O_m( params.outputCount )
        , O_m_w( params.outputCount,
	         params.neuronCount )
        , O_m_fb_w()
        , O_m_fb_w_scaling()

    {
        if ( params.inputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParams::inputCount must be not null" );
        if ( params.neuronCount <= 0 )
            throw std::invalid_argument(
                "NetworkParams::neuronCount must be not null" );
        if ( params.outputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParams::outputCount must be not null" );
	if ( !( params.leakingRate > 0.0 &&
                params.leakingRate <= 1.0 ) )
            throw std::invalid_argument(
                "NetworkParams::leakingRate must be within "
                "interval (0,1]" );
        if ( !( params.connectivity > 0.0f &&
                params.connectivity <= 1.0f ) )
            throw std::invalid_argument(
                "NetworkParams::connectivity must be within "
                "interval (0,1]" );

	matC in_m_w(params.neuronCount, params.inputCount, F_32);                                             //generate in_m_w matrix
	cv::RNG::fill(in_m_w, RNG::NORMAL, 0.f, params.deviation);                                           //fill matrix with random with a mean 0 and deviation .5



	/*
           a different method for fetching eigenvalues but is generally slower though more precise 
            
            porting to a reservoir generation function is intended for a future build

            cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, num_components);
            cv::Mat eigenvalues = num_components.eigenvalues.clone();
	*/
         // std::vector<float> rand_arr;

	do{
	   randomWeights = matC::create(params.neuronCount, params.neuronCount, F_32);
	   cv::RNG::fill(randomWeights, RNG::NORMAL, 0.f, params.deviation);                     
           cv::eigen(randomWeights, rand_arr);
	   rand_arr = cv::abs(rand_arr);
	}while( (rand_arr.at<float>(0,0)) > params.spectralRadius)


	  if ( params.deviation < 1.0f )              // if function is normalized already   //hint: '1' will get promoted to a float anyhow but why not typecast    
         {                                 
	    cv::SVD::compute(randomWeights,w,u,vt);
            E_m_w = u * vt.inv() ;                   //get identity / pseudoinverse matrix
	  }

	  
       else                    //basically this function is to normalize the function if the matrix is randomized for value beyond [1,-1] range
	  {
	    rand_arr = cv::abs(rand_arr);
	    double spectralRadius;
	    cv::minMaxLoc(rand_arr,NULL,&spectralRadius,NULL,NULL);           //eigenvalues are fetched then abs| | then maxcoeff
            E_m_w = ( randomWeights * params.spectralRadius / spectralRadius  ) ;         
        }
   
         in_m_scaling = matC::ones(params.inputCount,CV_8UC1);
         in_m_bias = matC::zeros(params.inputCount,CV_8UC1);
       	 O_m_w = matC::zeros(params.outputCount, params.neuronCount,CV_8UC1 );

	 
	 if (params.hasOutputFeedback)
        {
	    O_m_fb_w= matC::create(params.neuronCount, params.outputCount,CV_8UC1);
	    cv::RNG::fill(O_m_fb_w, RNG::NORMAL, 0.f, .5f);
	    O_m_fb_w_scaling = matC::ones(params.outputCount,1,CV_8UC1);
        }


	 //this allocates a matrix of leaking rates equal to  params.leakingRate
	 mLeakingRate = matC::create(params.neuronCount, params.neuronCount, CV_32FC2)* mParams.leakingRate; 
      	 mOneMinusLeakingRate = matC::ones(params.neuronCount,params.neuronCount, CV_32FC2) - mLeakingRate;


	 //since we are making an array,"std::vec" would be faster than "Mat" as "vec" allocates on the "stack" while "Mat" allocates on the heap but we would need then to overload operators for mat and vec interoperation or convert vec into mat each time with Mat(vec) function and this is a prototype/POC code so..... 

	 in_m = matC::zeros(params.inputCount,1,CV_8UC1);                          
	 E_m_activation = matC::create(params.inputCount,1,CV_8UC1);
	 cv::RNG::fill(E_m_activation,RNG::NORMAL,0.f,1);
	 O_m = matC::zeros(params.outputCount, 1, CV_8UC1);
	 
	 }

    Network::~Network()
    {
    }
}


 
