//
//  main.cpp
//  Lesson 3 - Kalman Filters
//
//  Created by Hadi Hajieghrary on 2/23/19.
//  Copyright © 2019 Hadi Hajieghrary. All rights reserved.
//

#include <iostream>
#include "Probability.hpp"
#include <time.h>
#include <algorithm>
#include <iomanip>
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

//template<typename T>
//void PlotDistribution(const Gaussian<T> G, int N = 10000, int B = 10);

void SimpleKalman(const MatrixXd& A, const MatrixXd& H, Eigen::Ref<MatrixXd> P, const MatrixXd& R,  Eigen::Ref<MatrixXd> X, const VectorXd& u, const VectorXd& z);
/*
     Xk = A Xk-1 + u
     zk = HXk
 
     MatrixXd A : State Transition Matrix
     MatrixXd H: Measurment Matrix
     MatrixXd P: Process Noise Covariance
     MatrixXd R: Measurment Noise Covariance
     VectorXd u: Control Command/Input
     VectorXd z: Measurments
 */


int main(int argc, const char * argv[]) {
  
 
    typedef Eigen::Matrix< double, 1, 1 > Vector1d;
    Vector1d y0;
    y0<<1;
    Vector1d y1;
    y1<<2;
    Vector1d y2;
    y2<<3;
    MatrixXd A (2,2);
    A<<1.0,1.0,0.0,1.0;
    MatrixXd H (1,2);
    H<<1.0,0.0;
    MatrixXd P (2,2);
    P<<1000.0,0.0,0.0,1000.0;
    MatrixXd R (1,1);
    R<<1.0;
    VectorXd X(2);
    X<<0.0,0.0;
    VectorXd u(2);
    u<<0.0,0.0;
    
    SimpleKalman(A, H, P, R, X, u, y0);
    cout<<"\nFirst Update:\n X=\n"<<X<<endl;
    cout<<"\n P =\n"<<P<<endl;
    
    SimpleKalman(A, H, P, R, X, u, y1);
    cout<<"\nSecond Update:\n X=\n"<<X<<endl;
    cout<<"\n P =\n"<<P<<endl;
    
    SimpleKalman(A, H, P, R, X, u, y2);
    cout<<"\nThird Update:\n X=\n"<<X<<endl;
    cout<<"\n P =\n"<<P<<endl;
    
    VectorXd mean = VectorXd(4);
    mean<<0.0 ,0.0 ,0.0 ,0.0;
    MatrixXd covar = MatrixXd(4,4);
    covar<<0.0, 0, 0 ,0,
           0, 0.0, 0, 0,
           0, 0, 0.0, 0,
           0, 0, 0, 0.0;
    
    Gaussian<MatrixXd> G(mean, covar);
    
    VectorXd V = G.draw();
    
    cout<<"Sample Vector: \n"<<V<<endl;
    
    return 0;
}


// Simple Kalman Filter Implementation
void SimpleKalman(const MatrixXd& A, const MatrixXd& H, Eigen::Ref<MatrixXd> P, const MatrixXd& R,  Eigen::Ref<MatrixXd> X, const VectorXd& u, const VectorXd& y){
    
    //Measurement
    VectorXd z = y - H*X;
    
    MatrixXd S = H * P * H.transpose() + R;
    MatrixXd K = P * H.transpose() * S.inverse() ;
    X = X + K*z;
    
    //Covariance Update
    P = ( MatrixXd::Identity(P.rows(),P.cols())- K*H ) * P;

    //Prediction
    X = A*X + u;
    P = A * P * A.transpose();
    
}


// Utility Functions
template<typename T>
void PlotDistribution(const Gaussian<T> G, int N, int B){
    
    //Odd number of Bins
    B = (B/2*2==B?B++:B);
    
    T* p = new T[N];
    p = G.draw(N);
    
    T a1 = G.Mean()- 2 * G.StandardDeviation();
    T a2 = G.Mean()+ 2 * G.StandardDeviation();
    
    T da = (a2 - a1) / (B - 2);
    
    int* Bins = new int[B];
    for (int i=0;i<B; i++) Bins[i] = 0;
    
    for (int i = 0; i<N; i++){
        if (p[i]<a1){
            Bins[0]++;
            continue;
        }
        if (p[i]>a2){
            Bins[B-1]++;
            continue;
        }
        for(int j=1; j<B-1;j++){
            if (p[i]>a1+(j-1)*da && p[i]<a1+j*da){
                Bins[j]++;
                break;
            }
        }
    }
    
    int MaxNum = *max_element(Bins, Bins+B);
    for(int i=0;i<B; i++)  Bins[i] = (double) Bins[i] / MaxNum * 20.0;
    
    for (int i=0; i<B; i++) printf("%03d ", Bins[i]);
    
    cout<<endl;
    
    for (int i=0;i<20;i++){
        for (int j=0;j<B;j++){
            if (Bins[j]>=(20-i)){
                cout<<" * "<<" ";
            }else{
                printf("%3s ", " ");
            }
        }
        cout<<endl;
        cout<< flush;
    }
    delete[] p;
    delete[] Bins;
    
}

