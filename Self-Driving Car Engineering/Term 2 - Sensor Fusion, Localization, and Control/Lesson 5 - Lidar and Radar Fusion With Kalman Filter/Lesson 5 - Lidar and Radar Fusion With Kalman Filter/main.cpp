//
//  main.cpp
//  Lesson 5 - Lidar and Radar Fusion With Kalman Filter
//
//  Created by Hadi Hajieghrary on 2/27/19.
//  Copyright © 2019 Hadi Hajieghrary. All rights reserved.
//

#include <iostream>
#include "Eigen/Dense"
#include "Probability.hpp"
#include "KalmanFilter.hpp"
#include <vector>

using namespace std;
using namespace Eigen;

//Kalman Filter Setup
    //States:
        //VectorXd X;
    //State Transition Matrix:
        //MatrixXd A;
    //Peocess Noise Matrix
        //MatrixXd Q;
    //State Estimation Covarience Matrix
        //MatrixXd P;
    //Control Input
        //VectorXd u;
    //Measurement:
        //VectorXd z;
    //Measurement Matrix:
        //MatrixXd H;
    //Measurement Noise Covarience Matrix
        //MatrixXd R;

class DynamicProcess{
    
private:
    
public:
    //State Vector
    VectorXd _X;
    //State Transixion Matrix
    MatrixXd _A;
    //Input (Control) Matrix
    MatrixXd _B;
    //Control Input
    VectorXd _U;
    //Process Noise Covarience Matrix
    MatrixXd _Q;
    
    //Output (Measurment/Observation)
    VectorXd _Y;
    //Output (Measurment/Observation) Matrix
    MatrixXd _C;
    //Measurement Noise Covarience Matrix
    MatrixXd _R;
    
    DynamicProcess(){};
    DynamicProcess(const MatrixXd& A, const MatrixXd& B, const MatrixXd& C);
    DynamicProcess(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& C, const MatrixXd& R);
    DynamicProcess(DynamicProcess& DP);
    DynamicProcess operator = (const DynamicProcess& DP);
    void Update(const VectorXd& U);
};

DynamicProcess::DynamicProcess(const MatrixXd& A, const MatrixXd& B, const MatrixXd& C):    _A(A),_B(B),_Q(MatrixXd::Zero(A.rows(), A.cols())), _C(C), _R(MatrixXd::Zero(C.rows(), C.rows())){
    if ( A.rows() != A.cols() ){
        throw "Error: Invalid State Transition Matrix A!";
    }
    if (A.rows() != B.rows()){
        throw "Error: Invalid Control Matrix!";
    }
    if (C.cols() != A.cols()){
        throw "Error: Invalid Observation Matrix!";
    }
}

DynamicProcess::DynamicProcess(const MatrixXd& A, const MatrixXd& B, const MatrixXd& Q, const MatrixXd& C, const MatrixXd& R):   _A(A),_B(B),_Q(Q), _C(C), _R(R){
    if ( A.rows() != A.cols() ){
        throw "Error: Invalid State Transition Matrix!";
    }
    if (A.rows() != B.rows()){
        throw "Error: Invalid Control Matrix!";
    }
    if (C.cols() != A.cols()){
        throw "Error: Invalid Observation Matrix!";
    }
    if ( Q.rows() != Q.cols() ){
        throw "Error: Invalid Process Noise Covarience Matrix!";
    }
    if ( A.rows() != Q.rows() ){
        throw "Error: Invalid Process Noise Covarience Matrix!";
    }
    if ( R.rows() != R.rows() ){
        throw "Error: Invalid Observation Noise Covarience Matrix!";
    }
    if ( R.rows() != C.rows() ){
        throw "Error: Invalid Observation Noise Covarience Matrix!";
    }
}

DynamicProcess::DynamicProcess(DynamicProcess& DP):_A(DP._A),_B(DP._B),_Q(DP._Q), _C(DP._C), _R(DP._R){
}

DynamicProcess DynamicProcess::operator = (const DynamicProcess& DP) {
    this->_A = DP._A;
    this->_B = DP._B;
    this->_Q = DP._Q;
    this->_C = DP._C;
    this->_R = DP._R;
    return *this;
}

void DynamicProcess::Update(const VectorXd& U){
    _U = U;
    Gaussian<MatrixXd> Q(MatrixXd::Zero(_Q.rows(),1), _Q);
    _X = _A * _X + _B * _U + Q.draw();
    Gaussian<MatrixXd> R(MatrixXd::Zero(_R.rows(),1), _R);
    _Y = _C * _X + R.draw();
}




int main(int argc, const char * argv[]) {
    
    const double dt= 0.01;
    //States:
    VectorXd X = VectorXd(4);
    X<<0, 0, 1, 1;
    //State Transition Matrix:
    MatrixXd A = MatrixXd(4,4);
    A<< 1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1;
    //Peocess Noise Matrix
    MatrixXd Q = MatrixXd(4,4);
    Q<< 0.1, 0,  0,  0,
        0,  0.1, 0,  0,
        0,  0,  0.1, 0,
        0,  0,  0,  0.1;
    //State Estimation Covarience Matrix
    MatrixXd P = MatrixXd(4,4);
    P<< 1000,   0,  0,  0,
        0,   1000,  0,  0,
        0,   0,  1000,  0,
        0,   0,  0,  1000;
    //Input Matrix
    MatrixXd B = MatrixXd(4,1);
    B<< 0,
        0,
        0,
        0;
    //Control Input
    VectorXd U = VectorXd(1);
    U<<1;
    //Measurement:
    VectorXd Y = VectorXd(4);
    Y<<1,   1,  0, 0;
    //Measurement Matrix:
    MatrixXd C = MatrixXd(4,4);
    C<< 1,  0,  0,  0,
        0,  1,  0,  0,
        0,  0,  1,  0,
        0,  0,  0,  1;
    //Measurement Noise Covarience Matrix
    MatrixXd R = MatrixXd(4,4);
    R<<1,   0,  0,  0,
       0,   1,  0,  0,
       0,   0,  1,  0,
       0,   0,  0,  0;
    DynamicProcess DP(A,B,Q,C,R);
    DP._X = X;
    DP.Update(U);
    cout<<"_X: "<<DP._X<<endl<<endl;
    
    DynamicModel DynModel(A,B,C,X);
    DynModel.Update(U);
    KalmanFilter Kalman(DynModel, Q, R);
  
    cout<<"Before Update: "<<endl<<endl;
    cout<<Kalman._X<<endl<<endl;
    cout<<Kalman._P<<endl<<endl;
    
    Kalman.Update(U, DP._Y);
    
    cout<<endl<<"After Updatr: "<<endl<<endl;
    cout<<Kalman._X<<endl<<endl;
    cout<<Kalman._P<<endl<<endl;
    
    return 0;
}
