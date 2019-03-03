//
//  probability.hpp
//  Lesson 3 - Kalman Filters
//
//  Created by Hadi Hajieghrary on 2/23/19.
//  Copyright © 2019 Hadi Hajieghrary. All rights reserved.
//

#ifndef probability_hpp
#define probability_hpp


#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

template<typename T>
class ProbabilityDistribution{
protected:
    ProbabilityDistribution(): _mean(NULL), _stddev(NULL), _var(NULL){};
    ~ProbabilityDistribution(){};
    T _mean;
    T _stddev;
    T _var;
    
public:
    
    virtual T draw() const = 0;
    virtual T* draw(int n) const = 0;
};


template<typename T>
class Gaussian: public ProbabilityDistribution<T>{
    
private:
    Gaussian(){};
    
public:
    Gaussian(T mean, T stddev);
    Gaussian(Gaussian& G);
    ~Gaussian();
    
    T Mean() const;
    T StandardDeviation() const;
    T Variance() const;
    // Product
    Gaussian<T> operator * (Gaussian<T> G);
    // Convolution
    Gaussian<T> operator || (Gaussian<T> G);
    
    T draw() const;
    T* draw(int n) const;
};


// Constructor
template<typename T>
Gaussian<T>::Gaussian(T mean, T stddev) {
    if (stddev < 0 ) throw "Standard Deviation Cannot be Negative!";
    this->_mean = mean;
    this->_stddev = stddev;
    this->_var = sqrt(this->_stddev);
}

// Copy Constructor
template<typename T>
Gaussian<T>::Gaussian(Gaussian& G){
    this->_mean = G._mean;
    this->_stddev = G._stddev;
    this->_var = sqrt(G._stddev);
}


//Destructor
template<typename T>
Gaussian<T>::~Gaussian(){
    
}


// Methods
template<typename T>
T Gaussian<T>::StandardDeviation() const{
    return this->_stddev;
}

template<typename T>
T Gaussian<T>::Mean() const{
    return this->_mean;
}

template<typename T>
T Gaussian<T>::Variance() const{
    return this->_var;
}

template<typename T>
T Gaussian<T>::draw() const{
    
    if (this->_stddev==0) return this->_mean;
    
    srand( (unsigned)time( NULL ) );
    
    T u1;
    T u2;
    u1 = (T) rand() / RAND_MAX;
    u2 = (T) rand() / RAND_MAX;
    
    T theta = 2*M_PI*u1;
    T R = sqrt(-2*log(u2));
    
    T G = this->_mean + this->_stddev * R * cos(theta);
    
    return G;
}

template<typename T>
T* Gaussian<T>::draw(int n) const{
    
    T* G = new T[n];
    
    if (this->_stddev==0){
        for (int i=0; i<n; i++) G[i] = this->_mean;
        return G;
    }
    
    
    srand( (unsigned)time( NULL ) );
    
    T* u1 = new T[n];
    T* u2 = new T[n];
    for(int i=0;i<n;i++){
        u1[i] = (T) rand() / RAND_MAX;
        u2[i] = (T) rand() / RAND_MAX;
    }
    
    T* theta = new T[n];
    T* R = new T[n];
    for(int i=0;i<n;i++){
        theta[i] = 2*M_PI*u1[i];
        R[i] = sqrt(-2*log(u2[i]));
    }
    
    for(int i=0;i<n;i++){
        G[i] = this->_mean + this->_stddev * R[i]*sin(theta[i]);
    }
    delete[] u1;
    delete[] u2;
    delete[] theta;
    delete[] R;
    
    return G;
}

// Operators
template<typename T>
Gaussian<T> Gaussian<T>::operator * (Gaussian<T> G){
    T _mean = ( pow(this->_stddev,2)*G._mean + pow(G._stddev,2)*this->_mean ) /
                                ( pow(this->_stddev,2) + pow(G._stddev,2) );
    T _stddev = sqrt( 1.0 / ( 1.0 / pow(this->_stddev,2) + 1.0 / pow(G._stddev,2) ) );
    
    Gaussian<T> N( _mean, _stddev );
    return N;
}


template<typename T>
Gaussian<T> Gaussian<T>::operator || (Gaussian<T> G){
    T _mean = ( G._mean + this->_mean );
    T _stddev = sqrt( pow(this->_stddev,2) +  pow(G._stddev,2) );
    
    Gaussian<T> N( _mean, _stddev );
    return N;
}


//Multi Variable Specialization of the Probability Classes

template<>
class ProbabilityDistribution<MatrixXd>{
protected:
    ProbabilityDistribution(){};
    ~ProbabilityDistribution(){};
    VectorXd _mean;
    MatrixXd _covar;

public:
    
    virtual VectorXd draw() const = 0;
    virtual VectorXd* draw(int n) const = 0;
};


template<>
class Gaussian<MatrixXd>: public ProbabilityDistribution<MatrixXd>{
    
private:
    Gaussian(){};
    
public:
    Gaussian(VectorXd mean, MatrixXd covar) {
        long intDim = mean.rows();
        if (covar.rows() != intDim || covar.cols()!=intDim)
            throw std::runtime_error("Error: The Mean and the Standard Deviation are not in Compatible Size!");
        Eigen::LLT<Eigen::MatrixXd> lltCov(covar); // Compute the Cholesky Decomposition of covar
        if(covar.determinant()!= 0 && lltCov.info() == Eigen::NumericalIssue)
            throw std::runtime_error("Covariance Matrix Should be Positive (Semi-)Definite!");
        this->_mean = mean;
        this->_covar = covar;
    }
    Gaussian(Gaussian& G){
        this->_mean = G._mean;
        this->_covar = G._covar;
    }
    ~Gaussian(){};
    
    VectorXd Mean() const{
        return this->_mean;
    }
    MatrixXd CoVariance() const{
        return this->_covar;
    }
    // Product
    Gaussian<MatrixXd> operator * (Gaussian<MatrixXd> G){
        VectorXd mean;
        MatrixXd covar;
        covar = (this->_covar).inverse() + (G._covar).inverse();
        covar = covar.inverse();
        mean = covar * (this->_covar*G._mean + G._covar*this->_mean);
        
        this->_mean = mean;
        this->_covar = covar;
        
        return *this;
    }
    // Convolution
    Gaussian<MatrixXd> operator || (Gaussian<MatrixXd> G){
        VectorXd mean;
        MatrixXd covar;
        covar = this->_covar + G._covar;
        mean = G._mean + this->_mean;
        
        this->_mean = mean;
        this->_covar = covar;
        
        return *this;
    }
    
    VectorXd draw() const{
        if ((this->_covar).determinant()==0) return this->_mean;
        
        srand( (unsigned)time( NULL ) );
        VectorXd Xd(_mean.rows());
        int n = 100;
        for (int i=0; i<n; i++){
            Xd+=(0.5 * MatrixXd::Random(_mean.rows(),1) + 0.5 * MatrixXd::Ones(_mean.rows(),1));
        }
        Xd = (Xd - MatrixXd::Ones(_mean.rows(),1)*(float)n/2)/sqrt((float)n/12);
        return _mean + _covar*Xd;
    }
    
    VectorXd* draw(int n) const{
        VectorXd* Res = new VectorXd[n];
        for (int i=0; i<n; i++){
            Res[i] = this->draw();
        }
        return Res;
    }
};



#endif /* probability_hpp */
