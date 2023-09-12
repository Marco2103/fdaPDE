#ifndef __STOCHASTIC_EDF_H__
#define __STOCHASTIC_EDF_H__

#include <random>
#include "../core/utils/Symbols.h"
#include "../core/NLA/SMW.h"
using fdaPDE::core::NLA::SMW;
#include "../models/regression/RegressionBase.h"
using fdaPDE::models::is_regression_model;
#include <Eigen/Cholesky>   // M aggiunto 


namespace fdaPDE {
namespace calibration{

  enum StochasticEDFMethod{
    WoodburyGCV,
    CholeskyGCV
  }; 

  // computes an approximation of the trace of S = \Psi*T^{-1}*\Psi^T*Q using a monte carlo approximation.
  template <typename Model>
  class StochasticEDF {
    static_assert(is_regression_model<Model>::value);

  private:
    Model& model_;       // ATT: anche qui tolto const 
    std::size_t r_;      // number of monte carlo realizations
    std::size_t seed_;
    DMatrix<double> Us_; // sample from Rademacher distribution
    DMatrix<double> Bs_; // \Psi^T*Q*Us_
    DMatrix<double> Y_;  // Us_^T*\Psi

    bool init_ = false;

    // StochasticEDFMethod method_;      // via?
    const unsigned int N_threshold = 3000;  // choose Wood/Chol depending on the problem size

  public:
    // // constructor
    // StochasticEDF(Model& model, std::size_t r, std::size_t seed, StochasticEDFMethod method = StochasticEDFMethod::WoodburyGCV)
    //   : model_(model), r_(r), seed_(seed), method_(method) {}
    // StochasticEDF(Model& model, std::size_t r, StochasticEDFMethod method = StochasticEDFMethod::WoodburyGCV)
    //   : StochasticEDF(model, r, std::random_device()(), method) {}

    // constructor
    StochasticEDF(Model& model, std::size_t r, std::size_t seed)
      : model_(model), r_(r), seed_(seed) {}
    StochasticEDF(Model& model, std::size_t r)
      : StochasticEDF(model, r, std::random_device()()) {}

    // ATT: abbiamo rimosso il qualifier "const" quando passa il modello (come in ExatEDF) perchè ora qui chiamiamo anche model_.Q()
    //      che deve poter calcolare Q_ e quindi modificare il modello (forse)
    

// evaluate trace of S exploiting a monte carlo approximation
    double compute() {
      std::size_t n = model_.Psi().cols(); // number of basis functions
      if(!init_){
	// compute sample from Rademacher distribution
	std::default_random_engine rng(seed_);
	std::bernoulli_distribution Be(0.5); // bernulli distribution with parameter p = 0.5
	Us_.resize(model_.n_obs(), r_); // preallocate memory for matrix Us
	// fill matrix
	for(std::size_t i = 0; i < model_.n_obs(); ++i){
	  for(std::size_t j = 0; j < r_; ++j){
	    if(Be(rng)) Us_(i,j) =  1.0;
	    else        Us_(i,j) = -1.0;
	  }
	}

	// prepare matrix Y 
	Y_ = Us_.transpose()*model_.Psi();
	init_ = true; // never reinitialize again
      }

      // prepare matrix Bs_
      Bs_ = DMatrix<double>::Zero(2*n, r_);
      if(!model_.hasCovariates()) {// non-parametric model
        Bs_.topRows(n) = - model_.PsiTD()*model_.W()*Us_;
      }
      else // semi-parametric model
        Bs_.topRows(n) = - model_.PsiTD()*model_.lmbQ(Us_);

      DMatrix<double> sol; // room for problem solution
      if(!model_.hasCovariates()){ // nonparametric case
        sol = model_.invA().solve(Bs_);
      }else{
        if (model_.n_basis() > N_threshold){ 
          // solve system (A+UCV)*x = Bs via woodbury decomposition using matrices U and V cached by model_
       
          sol = SMW<>().solve(model_.invA(), model_.U(), model_.XtWX(), model_.V(), Bs_);
        
        }
        else{   // Cholesky
          // solve system (A+UCV)*x = Bs via Cholesky factorization using matrices U and V cached by model_
          Eigen::LLT<DMatrix<double>> lltOfA; // compute the Cholesky decomposition of A
          lltOfA.compute( model_.T() );
          sol = lltOfA.solve(- Bs_.topRows(n));   
        }
        
      }
      // compute approximated Tr[S] using monte carlo mean
      double MCmean = 0;
      for(std::size_t i = 0; i < r_; ++i)
	MCmean += Y_.row(i).dot(sol.col(i).head(n));
      
      return MCmean/r_;
    }
  };

}}




  
#endif // __STOCHASTIC_EDF_H__
