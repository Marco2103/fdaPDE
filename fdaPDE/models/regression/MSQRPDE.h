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
    
    static const unsigned int h_;    // number of quantile orders 
    const std::vector<double> alphas_;                          // quantile order 
    // double rho_alpha(const double&) const;             // pinball loss function (quantile check function)

    // fdaPDE::SparseLU<SpMatrix<double>> invA_;         // factorization of matrix A      

    // DVector<double> py_{};                              // y - (1-2*alpha)|y - X*beta - f|
    // DVector<double> pW_{};                              // diagonal of W^k = 1/(2*n*|y - X*beta - f|)

    // // FPIRLS parameters (set to default)
    // std::size_t max_iter_ = 200;  
    // double tol_weights_ = 1e-6;  
    // double tol_ = 1e-6; 

    // Tolerances
    double gamma_ = 1; 
    double eps_ = 1e-6; 

    // algorithm's parameters 
    double tolerance_; 
    std::size_t max_iter_;
    std::size_t k_ = 0; // iteration index

    // System 
    SparseBlockMatrix<double,2,2> A_{}; // system matrix of non-parametric problem (2N x 2N matrix)
    fdaPDE::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
    DVector<double> b_{};

    // room for algorithm quantities 
    DVector<double> f_curr_{};     // estimate of the spatial field (1 x h*N vector)
    DVector<double> fn_curr_{};    // estimate of the spatial field (1 x h*n vector)
    DVector<double> g_curr_{};     // PDE misfit
    DVector<double> beta_curr_{};  // estimate of the coefficient vector (1 x h*q vector)
    DVector<double> f_prev_{};     // estimate of the spatial field (1 x h*N vector)
    DVector<double> fn_prev_{};    // estimate of the spatial field (1 x h*n vector)
    DVector<double> g_prev_{};     // PDE misfit
    DVector<double> beta_prev_{};  // estimate of the coefficient vector (1 x h*q vector)

    SpMatrix<double> Ih_;     // identity h x h 
    SpMatrix<double> In_;     // identity n x n
    SpMatrix<double> Iq_;     // identity q x q 
    SpMatrix<double> Ihn_;     // identity h*n x h*n 
    SpMatrix<double> Psi_multiple_{}; 
    SpMatrix<double> R0_multiple_{};
    SpMatrix<double> R1_multiple_{}; 
    SpMatrix<double> W_multiple_{}; 
    DMatrix<double> X_multiple_{};
    DMatrix<double> XtWX_multiple_{};
    Eigen::PartialPivLU<DMatrix<double>> invXtWX_multiple_{}; 
    SpMatrix<double> D_{};
    SpMatrix<double> D_script_{}; 
    DMatrix<double> Q_multiple_{}; 
    DMatrix<double> H_multiple_{}; 
    DMatrix<double> U_multiple_{};
    DMatrix<double> V_multiple_{};


  public:
    IMPORT_REGRESSION_SYMBOLS;

    std::vector<double> lambdas_;       // smoothing parameters in space   

    // constructor
    MSQRPDE() = default;
    MSQRPDE(const PDE& pde, std::vector<double>& alphas = {0.1, 0.5, 0.9}) : Base(pde), alphas_(alphas) {
      h_ = alphas_.size();
      f_curr_.resize(h_, n_basis());
      fn_curr_.resize(h_, n_obs());
      g_curr_.resize(h_, n_basis());
      beta_curr_.resize(h_, q());
      f_prev_.resize(h_, n_basis());
      fn_prev_.resize(h_, n_obs());
      g_prev_.resize(h_, n_basis());
      beta_prev_.resize(h_, q());
    }; 
    
    // setter
    // void setFPIRLSTolerance(double tol) { tol_ = tol; }
    // void setFPIRLSMaxIterations(std::size_t max_iter) { max_iter_ = max_iter; }
    // void setAlpha(const double &alpha) { alpha_ = alpha; }

    // ModelBase implementation
    void init_model() ;
    virtual void solve(); // finds a solution to the smoothing problem

    // required by FPIRLS (model_loss computes the unpenalized loss)
    // double model_loss(const DVector<double>& mu) const;

    // required by FPIRLS (initialize \mu for the first FPIRLS iteration)
    // DVector<double> initialize_mu() const;

    // required by FPIRLS (computes weight matrix and vector of pseudo-observations)
    // returns a pair of references to W^k and \tilde y^k
    // std::tuple<DVector<double>&, DVector<double>&> compute(const DVector<double>& mu);  


    bool crossing_constraints();

    void assemble_matrices(){

      Ih_.resize(h_, h_); 
	    Ih_.setIdentity();

      In_.resize(n_obs(), n_obs()); 
	    In_.setIdentity();

      Iq_.resize(n_obs(), n_obs()); 
	    Iq_.setIdentity();

      Ihn_.resize(h_*n_obs(), h_*n_obs()); 
	    Ihn_.setIdentity();

      // Assemble FEM, mass and stiffness matrices
      Psi_multiple_ = Kronecker(Ih_, Psi()); 
      R0_multiple_ = Kronecker(DiagMatrix<double>(lambdas_), R0());
      R1_multiple_ = Kronecker(DiagMatrix<double>(lambdas_), R1());

      X_multiple_ = Kronecker(Ih_, X()); 

      BandMatrix<double> E_(h_-1, h_, 1, 1);
      E_.diagonal(1).setConstant(1);
      E_.diagonal(-1).setConstant(-1);

      D_ = Kronecker(SpMatrix<double>(E_), Iq_);
      D_script_ = Kronecker(SpMatrix<double>(E_), In_); 

      

    }


    // // iGCV interface implementation
    // virtual const DMatrix<double>& T();  
    // virtual const DMatrix<double>& Q(); 

    // // returns the euclidian norm of y - \hat y
    // double norm(const DMatrix<double>& obs, const DMatrix<double>& fitted) const ; 

    // getters
    // const fdaPDE::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    // const DMatrix<double>& U() const { return U_; }
    // const DMatrix<double>& V() const { return V_; }


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