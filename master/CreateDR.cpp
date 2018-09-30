


void CreateDr(unsigned neuronCount,
	      float deviation,
              float spectralRadius,
	      const matC & E_m_w)
    {
	do{
	   randomWeights = matC::create(neuronCount, neuronCount, F_32);
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
	    double wspectralRadius;
	    cv::minMaxLoc(rand_arr,NULL,&spectralRadius,NULL,NULL);           //eigenvalues are fetched then abs| | then maxcoeff
            E_m_w = ( randomWeights * params.spectralRadius / spectralRadius  ) ;         
        }

      if(bidir
