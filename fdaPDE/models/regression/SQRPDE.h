#ifndef __SQRPDE_H__
#define __SQRPDE_H__

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
#include "../../calibration/iGCV.h"
using fdaPDE::calibration::iGCV;
// regression module imports
#include "../SamplingDesign.h"
#include "RegressionBase.h"
using fdaPDE::models::RegressionBase;
#include "FPIRLS.h"
using fdaPDE::models::FPIRLS ; 

// For log1exp
// #include<Rcpp.h>
// #include<Rmath.h>

namespace fdaPDE{
namespace models{
  
  template <typename PDE, typename SamplingDesign>
  class SQRPDE : public RegressionBase<SQRPDE<PDE, SamplingDesign>>, public iGCV {
    // compile time checks
    static_assert(std::is_base_of<PDEBase, PDE>::value);
  private:
    typedef RegressionBase<SQRPDE<PDE, SamplingDesign>> Base;
    
    double alpha_;                                          // quantile order 
    double rho_alpha(const double&) const;  // pinball loss function (quantile check function)

    // SpMatrix<double> A_{};                         // system matrix of non-parametric problem (2N x 2N matrix)
                                                      // non è usata da nessuna parte, va salvata comunque o la togliamo?

    fdaPDE::SparseLU<SpMatrix<double>> invA_;         // factorization of matrix A

    // Commento questi membri così stiamo usando quelli di RegressionBase
    // DiagMatrix<double> W_{};                            // weight matrix at FPRILS convergence
    // DMatrix<double> XtWX_{}; 
    // Eigen::PartialPivLU<DMatrix<double>> invXtWX_{};  // factorization of the dense q x q matrix XtWX_

    DVector<double> py_{};                              // y - (1-2*alpha)|y - X*beta - f|
    DVector<double> pW_{};                              // diagonal of W^k = 1/(2*n*|y - X*beta - f|)

    // FPIRLS parameters (set to default)
    std::size_t max_iter_ = 200;  
    double tol_weights_; 
    double tol_;

    // Debug
    double Jfinal_sqrpde_;
    std::size_t niter_sqrpde_;

    // matrices related to woodbury decomposition -> tolte perchè lui le ha aggiunte in RegressionBase
    // DMatrix<double> U_{};
    // DMatrix<double> V_{};  

  public:
    IMPORT_REGRESSION_SYMBOLS;
    using Base::lambdaS; // smoothing parameter in space
    // constructor
    SQRPDE() = default;
    SQRPDE(const PDE& pde, double alpha = 0.5) : Base(pde), alpha_(alpha) {};   

    
    // setter
    void setFPIRLSTolerance(double tol) { tol_ = tol; }
    void setFPIRLSMaxIterations(std::size_t max_iter) { max_iter_ = max_iter; }

    // ModelBase implementation
    void init_model() { return; }
    virtual void solve(); // finds a solution to the smoothing problem

    // required by FPIRLS (computes weight matrix and vector of pseudo-observations)
    // returns a pair of references to W^k and \tilde y^k
    std::tuple<DVector<double>&, DVector<double>&> compute(const DVector<double>& mu);

    // model_loss computes the unpenalized loss (it is called by FPIRLS)
    double model_loss(const DVector<double>& mu); // private or public?

    DVector<double> initialize_mu() const ; 

    
    // iGCV interface implementation
    virtual const DMatrix<double>& T(); // T = A 
    virtual const DMatrix<double>& Q(); 

    // returns the euclidian norm of y - \hat y
    double norm(const DMatrix<double>& obs, const DMatrix<double>& fitted) const ; 

    // getters
    // const DiagMatrix<double>& W() const { return W_; } 
    // const DMatrix<double>& XtWX() const { return XtWX_; }
    // const Eigen::PartialPivLU<DMatrix<double>>& invXtWX() const { return invXtWX_; }
    
    // const DVector<double>& py() const { return py_; }    debug

    // const SpMatrix<double>& A() const { return A_; }

    const fdaPDE::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    const DMatrix<double>& U() const { return U_; }
    const DMatrix<double>& V() const { return V_; }
    const bool massLumpingGCV() const { return Base::massLumpingGCV(); }  // necessario perchè 
    // altrimenti in GCV.h non posso fare model_.massLumping() 
    // no ref perchè è ritorno un temporary object

    // Debug
    const double& J_final_sqrpde() const { return Jfinal_sqrpde_; } 
    const std::size_t& niter_sqrpde() const { return niter_sqrpde_; } 
    void setTolerances(double tol_weigths, double tol_FPIRLS) { 
      tol_weights_ = tol_weigths; 
      tol_ = tol_FPIRLS; 
    }
    const DiagMatrix<double> lumped_invR0() const { return lumped_invR0_; }
    


    virtual ~SQRPDE() = default;
  };
 

 template <typename PDE_, typename SamplingDesign_>
  struct model_traits<SQRPDE<PDE_, SamplingDesign_>> {
    typedef PDE_ PDE;
    typedef SpaceOnly regularization;
    typedef SamplingDesign_ sampling;
    typedef MonolithicSolver solver;
    static constexpr int n_lambda = 1;
  };

  // sqrpde trait
  template <typename Model>
  struct is_sqrpde { static constexpr bool value = is_instance_of<Model, SQRPDE>::value; };

#include "SQRPDE.tpp"
}}
    
#endif // __SQRPDE_H__
