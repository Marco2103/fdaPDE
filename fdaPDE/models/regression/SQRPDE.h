#ifndef __SQRPDE_H__
#define __SQRPDE_H__

#include <memory>
#include <type_traits>
// CORE imports
#include "../../core/utils/Symbols.h"
#include "../../core/FEM/PDE.h"
using fdaPDE::core::FEM::PDEBase;
#include "../ModelBase.h"
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

namespace fdaPDE{
namespace models{
  
  template <typename PDE, Sampling SamplingDesign>
  class SQRPDE : public RegressionBase<SQRPDE<PDE, SamplingDesign>>, public iGCV {
    // compile time checks
    static_assert(std::is_base_of<PDEBase, PDE>::value);
  private:
    typedef RegressionBase<SQRPDE<PDE, SamplingDesign>> Base;
    
    double alpha_;                                          // quantile order 
    double rho_alpha(const double&) const;  // pinball loss function (quantile check function)

    SpMatrix<double> A_{};                          // system matrix of non-parametric problem (2N x 2N matrix)
    fdaPDE::SparseLU<SpMatrix<double>> invA_{};         // factorization of matrix A

    DiagMatrix<double> W_{};                            // weight matrix at FPRILS convergence 
    DMatrix<double> XtWX_{}; 
    Eigen::PartialPivLU<DMatrix<double>> invXtWX_{};  // factorization of the dense q x q matrix XtWX_

    DVector<double> py_{};                            // y - (1-2*alpha)|y - X*beta - f|
    DVector<double> pW_{};                              // diagonal of W^k = 1/(2*n*|y - X*beta - f|)

    // FPIRLS parameters (set to default)
    std::size_t max_iter_ = 15;
    double tol_ = 0.0002020;

    // matrices related to woodbury decomposition
    DMatrix<double> U_{};
    DMatrix<double> V_{};  

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

    double compute_J_unpenalized(const DVector<double>& mu); // private or public?

    DVector<double> initialize_mu() const ; 

    
    // iGCV interface implementation
    virtual const DMatrix<double>& T(); // T = A 
    virtual const DMatrix<double>& Q(); 

    // returns the euclidian norm of y - \hat y
    double norm(const DMatrix<double>& obs, const DMatrix<double>& fitted) const ; 

    // getters
    const DiagMatrix<double>& W() const { return W_; } 
    const DMatrix<double>& XtWX() const { return XtWX_; }
    const SpMatrix<double>& A() const { return A_; }
    const fdaPDE::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    const DMatrix<double>& U() const { return U_; }
    const DMatrix<double>& V() const { return V_; }

    
    virtual ~SQRPDE() = default;
  };
  
  template <typename PDE_, Sampling SamplingDesign>
  struct model_traits<SQRPDE<PDE_, SamplingDesign>> {
    typedef PDE_ PDE;
    typedef SpaceOnly RegularizationType;
    static constexpr Sampling sampling = SamplingDesign;
    static constexpr SolverType solver = SolverType::Monolithic;
    static constexpr int n_lambda = 1;
  };
 

#include "SQRPDE.tpp"
}}
    
#endif // __SQRPDE_H__
