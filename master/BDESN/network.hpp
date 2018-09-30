#ifndef __BDESN_NETWORK_HPP__
#define __BDESN_NETWORK_HPP__

#include <bdesn/export.h>
#include <memory>
#include <cmath>
#include <cstring>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <bdesn/exceptions.hpp>
#include <bdesn/network.h>
#include <bdesn/network.hpp>
#include <network_nsli.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>  

namespace BDESN {
    #define F_32 CV_32FC3 
    typedef cv::Mat matC;
    
  class Network;

    struct NetworkParams
    {
        unsigned inputCount;
        unsigned neuronCount;
        float deviation;                         
        unsigned outputCount;
        float leakingRate;
        float spectralRadius;
        float connectivity;
        bool linearOutput;
        float ForgettingFactor;
        float InitialCovariance;
        bool hasOutputFeedback;
        matC rand_arr, randomWeights,w,u,vt;
  

        NetworkParams()
            : inputCount( 0 )
            , neuronCount( 0 )
            , deviation( 0 )
            , outputCount( 0 )
            , leakingRate( 1.0f )
            , spectralRadius( 1.0f )
            , connectivity( 1.0f )
            , linearOutput( false )
            , onlineTrainingForgettingFactor( 1.0f )
            , onlineTrainingInitialCovariance( 1000.0f )
            , hasOutputFeedback(true)
        {}
    };

  BDESN_EXPORT std::unique_ptr< Network > CreateNetwork( const NetworkParams & );
  BDESN_EXPORT void Network::SetInputScalings( const cv::Mat & scalings );
  BDESN_EXPORT void Network::SetInputBias( const cv::Mat & bias );
  BDESN_EXPORT void Network::SetFeedbackScalings(const cv::Mat & scalings );
  BDESN_EXPORT void Network::CaptureOutput(cv::Mat  & output );
  BDESN_EXPORT void Network::CaptureActivations(cv::Mat & activations );
  BDESN_EXPORT void Network::CaptureTransformedInput(cv::Mat & input );
  BDESN_EXPORT void Network::Train(const cv::Mat & inputs,const cv::Mat & outputs );
  BDESN_EXPORT void Network::SetInputs( const cv::Mat & inputs );
  BDESN_EXPORT void Network::Step( float step );     //learning function
  


} 

#endif 
