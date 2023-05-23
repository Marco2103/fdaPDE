#ifndef __FPIRLS_H__
#define __FPIRLS_H__

#include "../../core/utils/Symbols.h"
#include "../../core/NLA/SparseBlockMatrix.h"
using fdaPDE::core::NLA::SparseBlockMatrix;
#include "../../core/NLA/SMW.h"
using fdaPDE::core::NLA::SMW;
#include "Distributions.h"
#include <cstddef>
#include "../ModelTraits.h"
#include "SRPDE.h"
#include "STRPDE.h"

#include <chrono>

namespace fdaPDE{
namespace models{

  // trait to select model type to use in the internal loop of FPIRLS
  template <typename Model>
  struct FPIRLS_internal_solver {
    typedef typename std::decay<Model>::type Model_;
    using type = typename std::conditional<
      !is_space_time<Model_>::value,
      // space-only problem
      SRPDE <typename model_traits<Model_>::PDE, model_traits<Model_>::sampling>,
      // space-time problem
      STRPDE<typename model_traits<Model_>::PDE, typename model_traits<Model_>::RegularizationType,
	     model_traits<Model_>::sampling, model_traits<Model_>::solver>
      >::type;
  };
  
  // a general implementation of the Functional Penalized Iterative Reweighted Least Square (FPIRLS) algorithm
  template <typename Model, typename Distribution = Gaussian>
  class FPIRLS {
  private:
    typedef typename std::decay<Model>::type Model_;
    // data characterizing the behaviour of the algorithm
    Distribution distribution_{};
    Model& m_;
    // algorithm's parameters 
    double tolerance_; 
    std::size_t max_iter_;
    std::size_t k_ = 0; // FPIRLS iteration index

    // define internal problem solver and initialize it
    typename FPIRLS_internal_solver<Model>::type solver_;
    
    DVector<double> mu_{};    // \mu^k = [ \mu^k_1, ..., \mu^k_n ] : mean vector at step k
    // parameters at convergece
    DVector<double> f_{};     // estimate of non-parametric spatial field
    DVector<double> g_{};     // PDE misfit
    DVector<double> beta_{};  // estimate of coefficient vector
    DVector<double> W_{};     // weight matrix

    DVector<double> mu_init{};    // per debug -> da togliere 
    DMatrix<double> matrix_pseudo{};
    DMatrix<double> matrix_weight{}; 
    DMatrix<double> matrix_beta{};
    DMatrix<double> matrix_f{}; 

  public:
    // constructor
    FPIRLS(const Model& m, double tolerance, std::size_t max_iter)
      : m_(m), tolerance_(tolerance), max_iter_(max_iter) {
        // Define solver 
        if constexpr(!is_space_time<Model>::value) // space-only
	        solver_ = typename FPIRLS_internal_solver<Model>::type(m_.pde());
        else{ // space-time
	        solver_ = typename FPIRLS_internal_solver<Model>::type(m_.pde(), m_.time_domain());
	      // in case of parabolic regularization derive initial condition from input model
	        if constexpr(std::is_same<typename model_traits<Model_>::RegularizationType,       // regularization,
		        SpaceTimeParabolic>::value)
	            solver_.setInitialCondition(m_.s());
        }
        // Initialize solver
        solver_.setLambda(m_.lambda());
        solver_.init_pde();
        // prepare data for solver_, copy covariates if present
        solver_.data() = m_.data();
        solver_.init_regularization();
        solver_.init_sampling();

        // prepare data for solver, copy covariates if present
      // BlockFrame<double, int> df = m_.data();
      // if(m_.hasCovariates()) df.insert<double>(DESIGN_MATRIX_BLK, m_.X()); 

        matrix_pseudo.resize(m_.n_obs() , max_iter_); 
        matrix_weight.resize(m_.n_obs() , max_iter_);
        matrix_beta.resize(m_.q() , max_iter_);
        matrix_f.resize(m_.n_obs() , max_iter_);

      };
    
    // executes the FPIRLS algorithm
    void compute() {

      static_assert(is_regression_model<Model>::value);  
      mu_ = m_.initialize_mu(); 

      mu_init = mu_ ;    // da togliere
  
      distribution_.preprocess(mu_);
          
      // algorithm stops when an enought small difference between two consecutive values of the J is recordered
      double J_old = tolerance_+1; double J_new = 0;

      // start loop
      while(k_ < max_iter_ && std::abs(J_new - J_old) > tolerance_){
      //   while(k_ < max_iter_ && std::abs(J_new - 0.04472646666589305) > 1e-3){  --> to check a specific value of J 
	// request weight matrix W and pseudo-observation vector \tilde y from model --> !!!!

  // std::cout << "k in FPRLS is: " << k_ << std::endl ; 
	auto pair = m_.compute(mu_);    // aggiunto un k in input 
	// solve weighted least square problem
	// \argmin_{\beta, f} [ \norm(W^{1/2}(y - X\beta - f_n))^2 + \lambda \int_D (Lf - u)^2 ]

	solver_.data().template insert<double>(OBSERVATIONS_BLK, std::get<1>(pair));
	solver_.data().template insert<double>(WEIGHTS_BLK, std::get<0>(pair));
  // std::cout << (std::get<1>(pair)).squaredNorm() << std::endl; 
  matrix_pseudo.col(k_) = std::get<1>(pair) ; 
  matrix_weight.col(k_) = std::get<0>(pair); //.diagonal() ; 
	// update solver_ to change in the weight matrix

	solver_.update_to_data();

	solver_.init_model(); 

	solver_.solve();
	// extract estimates from solver

	f_ = solver_.f(); g_ = solver_.g();
  // matrix_f.col(k_) = f_ ; 

	if(m_.hasCovariates()) {
    beta_ = solver_.beta();
    matrix_beta.col(k_) = beta_ ; 
  }
	
	// update value of \mu_
	DVector<double> fitted = solver_.fitted(); // compute fitted values

	mu_ = distribution_.inv_link(fitted);

	// compute value of functional J for this pair (\beta, f): \norm{V^{-1/2}(y - \mu)}^2 + \int_D (Lf-u)^2 
  double J = m_.compute_J_unpenalized(mu_) + m_.lambdaS()*g_.dot(m_.R0()*g_); // aggiunto il lambda
  // compute_J_unpen -> model_loss
	// prepare for next iteration

	k_++; J_old = J_new; J_new = J;


      }

  if (k_ == max_iter_)
    std::cout << "MAX ITER RAGGIUNTO " << std::endl;  

  std::cout << "Number of FPIRLS iterations: " << k_ << std::endl;
  std::cout << "Value of J at the last iteration: " <<  std::setprecision(16) << J_new << std::endl;  

      // store weight matrix at convergence
      W_ = std::get<0>(m_.compute(mu_));    
    

      return;
    } 

    // getters 
    const DVector<double>& weights() const { return W_; }                               // weights matrix W at convergence
    const DVector<double>& beta() const { return beta_; }                               // estimate of coefficient vector 
    const DVector<double>& f() const { return f_; }                                     // estimate of spatial field 
    const DVector<double>& g() const { return g_; }                                     // PDE misfit
    std::size_t n_iter() const { return k_ - 1; }                                       // number of iterations
    const typename FPIRLS_internal_solver<Model>::type & solver() const { return solver_; }   // solver  

    const DVector<double>& mu_initialized() const { return mu_init; }    // per debug -> da togliere
    const DMatrix<double>& matrix_pseudo_fpirls() const { return matrix_pseudo; }
    const DMatrix<double>& matrix_weight_fpirls() const { return matrix_weight; } 
    const DMatrix<double>& matrix_beta_fpirls() const { return matrix_beta; } 
    const DMatrix<double>& matrix_f_fpirls() const { return matrix_f; } 
  };
  
}}



#endif // __FPIRLS_H__
