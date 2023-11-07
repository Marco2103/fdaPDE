#ifndef __MSQRPDE_H__
#define __MSQRPDE_H__

#include <memory>
#include <type_traits>
// CORE imports
#include "../../core/utils/Symbols.h"
#include "../../core/FEM/PDE.h"
using fdaPDE::core::FEM::PDEBase;
#include "../ModelBase.h"
#include "../ModelTraits.h"
#include "../ModelMacros.h"
#include "../../core/NLA/SparseBlockMatrix.h"
using fdaPDE::core::NLA::SparseBlockMatrix;
#include "../../core/NLA/SMW.h"
using fdaPDE::core::NLA::SMW;
// calibration module imports
// #include "../../calibration/iGCV.h"
// using fdaPDE::calibration::iGCV;
// regression module imports
#include "../SamplingDesign.h"
#include "RegressionBase.h"
using fdaPDE::models::RegressionBase;
#include "FPIRLS.h"
using fdaPDE::models::FPIRLS; 

#include "SQRPDE.h"
#include "../../core/utils/DataStructures/BlockVector.h"
using fdaPDE::BlockVector;
using Eigen::internal::BandMatrix;


namespace fdaPDE{
namespace models{
	    
  
  template <typename PDE, typename SamplingDesign>
  class MSQRPDE : public RegressionBase<MSQRPDE<PDE, SamplingDesign>> {  // , public iGCV {
    // compile time checks
    static_assert(std::is_base_of<PDEBase, PDE>::value);
  private:
    typedef RegressionBase<MSQRPDE<PDE, SamplingDesign>> Base;
    
    unsigned int h_;                     // number of quantile orders 
    const std::vector<double> alphas_;   // quantile order 

    // algorithm's parameters 
    double gamma0_ = 1.0;                  // crossing penalty 
    double eps_ = 1e-6;                    // crossing tolerance 
    double C_ = 1.5;                       // crossing penalty factor
    double tolerance_ = 1e-5;              // convergence tolerance 
    double tol_weights_ = 1e-6;            // weights tolerance
    std::size_t max_iter_ = 50;            // inner max number of iterations 
    std::size_t max_iter_global_ = 100;    // outer max number of iterations 
    std::size_t k_ = 0;                    // inner iteration index
    std::size_t iter_ = 0;                 // outer iteration index

    bool debug_ = false; 

    // linear system  
    SparseBlockMatrix<double,2,2> A_{};         // system matrix of non-parametric problem (2hN x 2hN matrix)
    fdaPDE::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
    DVector<double> b_{};                       // system rhs 

    // room for solution 
    DVector<double> f_curr_{};     // current estimate of the spatial field f (1 x h*N vector)
    DVector<double> fn_curr_{};    // current estimate of the spatial field f_n (1 x h*n vector)
    DVector<double> g_curr_{};     // current PDE misfit
    DVector<double> beta_curr_{};  // current estimate of the coefficient vector (1 x h*q vector)

    DVector<double> f_init_{}; 
    DVector<double> fn_init_{}; 
    DVector<double> g_init_{}; 
    DVector<double> beta_init_{}; 

    // room for algorithm quantities 
    SpMatrix<double> Ih_;                 // identity h x h 
    SpMatrix<double> In_;                 // identity n x n
    SpMatrix<double> Iq_;                 // identity q x q 
    SpMatrix<double> Ihn_;                // identity h*n x h*n 
    SpMatrix<double> D_{};                
    SpMatrix<double> D_script_{}; 
    DVector<double> l_hn_{}; 
    SpMatrix<double> Psi_multiple_{}; 
    SpMatrix<double> R0_multiple_{};
    SpMatrix<double> R1_multiple_{}; 
    DVector<double> z_{};
    DVector<double> w_{};
    DiagMatrix<double> W_bar_{};
    SpMatrix<double> W_multiple_{}; 
    DMatrix<double> X_multiple_{};
    DMatrix<double> XtWX_multiple_{};
    Eigen::PartialPivLU<DMatrix<double>> invXtWX_multiple_{}; 
    DMatrix<double> Q_multiple_{}; 
    DMatrix<double> H_multiple_{}; 
    DMatrix<double> U_multiple_{};
    DMatrix<double> V_multiple_{};

    // debug 
    DiagMatrix<double> W_bar_debug{}; 
    DiagMatrix<double> Delta_debug{}; 
    SpMatrix<double> Dscriptj_debug{}; 
    DMatrix<double> XtWX_multiple_debug{}; 
    SpMatrix<double> W_multiple_debug{};
    DMatrix<double> H_multiple_debug{};


  public:
    IMPORT_REGRESSION_SYMBOLS;

    DVector<double> lambdas_;       // smoothing parameters in space   

    // constructor
    MSQRPDE() = default;
    MSQRPDE(const PDE& pde, std::vector<double>& alphas = {0.1, 0.5, 0.9}) : Base(pde), alphas_(alphas) {
      // // Check if the provided quantile orders are an increasing sequence 
      // auto i = std::adjacent_find(alphas.begin(), alphas.end(), std::greater_equal<int>());
      // assert(i == alphas.end(), "Quantile orders must be an increasing sequence"); 

      h_ = alphas_.size();
    }; 

    // getters
    const DVector<double>& f() const { return f_curr_; };            // estimate of spatial field
    const DVector<double>& g() const { return g_curr_; };            // PDE misfit
    const DVector<double>& beta() const { return beta_curr_; };      // estimate of regression coefficients
    const SparseBlockMatrix<double,2,2>& A_mult() const { return A_; };    // debug
    const SpMatrix<double>& Psi_mult() const { return Psi_multiple_; };    // debug
    const SpMatrix<double>& W_mult() const { return W_multiple_debug; };        // debug
    const SpMatrix<double>& D_script() const { return D_script_; };       // debug  
    const DiagMatrix<double>& Delta_mult() const { return Delta_debug; };         // debug
    const DiagMatrix<double>& Wbar_mult() const { return W_bar_debug; };         // debug 
    const DMatrix<double>& Q_mult() const { return Q_multiple_; };         // debug
    const DMatrix<double>& H_mult_debug() const { return H_multiple_debug; };         // debug 
    const DMatrix<double>& X_mult() const { return X_multiple_; };         // debug 
    const SpMatrix<double>&  Dscriptj() const { return Dscriptj_debug; };       // debug 
    const DMatrix<double>& XtWX_multiple() const { return XtWX_multiple_debug; };  // debug

    // ModelBase implementation
    void init_model() ;
    virtual void solve(); // finds a solution to the smoothing problem

    const bool crossing_constraints() const;

    void setLambdas_S(DMatrix<double> l){
      lambdas_.resize(l.rows()); 
      for(auto i = 0; i < l.rows(); ++i)
         lambdas_(i) = l(i,0);  
    }
       
    void assemble_matrices(){

      if(debug_)
        std::cout << "MSQRPDE assemble: here 1" << std::endl;

      // room for solution 
      f_curr_.resize(h_*n_basis());
      fn_curr_.resize(h_*n_obs());
      g_curr_.resize(h_*n_basis());
      f_init_.resize(h_*n_basis());
      fn_init_.resize(h_*n_obs());
      g_init_.resize(h_*n_basis());
    
      if(hasCovariates()){
        beta_curr_.resize(h_*q());
        beta_init_.resize(h_*q());
      }


      // set identity matrices 
      Ih_.resize(h_, h_); 
	    Ih_.setIdentity();
      In_.resize(n_obs(), n_obs()); 
	    In_.setIdentity();
      Iq_.resize(q(), q()); 
	    Iq_.setIdentity();
      Ihn_.resize(h_*n_obs(), h_*n_obs()); 
	    Ihn_.setIdentity();

      if(debug_)
        std::cout << "MSQRPDE assemble: here 4" << std::endl;

      // assemble FEM, mass and stiffness matrices
      Psi_multiple_ = Kronecker(Ih_, Psi()); 
      R0_multiple_ = Kronecker(SpMatrix<double>(DiagMatrix<double>(lambdas_)), R0());
      R1_multiple_ = Kronecker(SpMatrix<double>(DiagMatrix<double>(lambdas_)), R1());

      if(debug_)
        std::cout << "MSQRPDE assemble: here 5" << std::endl;

      // assemble
      if(hasCovariates()){ 
        X_multiple_ = Kronecker(DMatrix<double>(Ih_), X()); 
        XtWX_multiple_.resize(h_*q(), h_*q()); 
        U_multiple_.resize(2*h_*n_basis(), h_*q());
        V_multiple_.resize(h_*q(), 2*h_*n_basis());
      }

      // assemble l_hn_
      l_hn_.resize(h_*n_obs()); 
      l_hn_ = DVector<double>::Zero(h_*n_obs());
      l_hn_.block(0,0, n_obs(), 1) = -DVector<double>::Ones(n_obs());
      l_hn_.block((h_-1)*n_obs(),0, n_obs(), 1) = DVector<double>::Ones(n_obs());

      // BandMatrix<double> E_(h_-1, h_, 1, 1);
      // E_.diagonal(1).setConstant(1);
      // E_.diagonal(-1).setConstant(-1);
      SpMatrix<double> E_{};
      E_.resize(h_-1, h_); 
      // E_.reserve(2*h_-1); 

      std::vector<fdaPDE::Triplet<double>> tripletList;
      tripletList.reserve(2*(h_-1));

      for(std::size_t i = 0; i < h_-1; ++i){
	      tripletList.emplace_back(i, i+1, 1.0);
        tripletList.emplace_back(i, i, -1.0);
      }

      E_.setFromTriplets(tripletList.begin(), tripletList.end());
      E_.makeCompressed();

      if(debug_)
        std::cout << "MSQRPDE assemble: here 8" << std::endl;

      if(hasCovariates()){
        D_ = Kronecker(E_, Iq_); 
      }       
      D_script_ = Kronecker(E_, In_);  

      if(debug_)
        std::cout << "MSQRPDE assemble: here 9" << std::endl;

    }

    DVector<double> rho_alpha(const double&, const DVector<double>&) const; 
    DVector<double> fitted(unsigned int j) const; 
    DVector<double> fitted() const; 
    void abs_res_adj(DVector<double>& w); 
    double model_loss() const; 
    double crossing_penalty() const;
    double crossing_penalty_f() const; 
    double crossing_penalty_param() const; 

    // // iGCV interface implementation
    // virtual const DMatrix<double>& T();  
    // virtual const DMatrix<double>& Q(); 

    const DMatrix<double>& H_multiple(); 
    const DMatrix<double>& Q_multiple(); 

    virtual ~MSQRPDE() = default;
  };
 

 template <typename PDE_, typename SamplingDesign_>
  struct model_traits<MSQRPDE<PDE_, SamplingDesign_>> {
    typedef PDE_ PDE;
    typedef SpaceOnly regularization;
    typedef SamplingDesign_ sampling;
    typedef MonolithicSolver solver;
    static constexpr int n_lambda = 1;
  };

  // sqrpde trait
  template <typename Model>
  struct is_msqrpde { static constexpr bool value = is_instance_of<Model, MSQRPDE>::value; };

#include "MSQRPDE.tpp"
}}
    
#endif // __MSQRPDE_H__