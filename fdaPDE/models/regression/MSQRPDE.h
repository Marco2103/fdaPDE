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
    double gamma_ = 1;               // crossing penalty 
    double eps_ = 1e-6;              // crossing tolerance 
    double C_ = 5;                   // crossing penalty factor
    double tolerance_ = 1e-6;        // convergence tolerance 
    std::size_t max_iter_ = 200;     // max number of iterations 
    std::size_t k_ = 0;              // iteration index

    // linear system  
    SparseBlockMatrix<double,2,2> A_{};         // system matrix of non-parametric problem (2hN x 2hN matrix)
    fdaPDE::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
    DVector<double> b_{};                       // system rhs 

    // room for solution 
    DVector<double> f_curr_{};     // current estimate of the spatial field f (1 x h*N vector)
    DVector<double> fn_curr_{};    // current estimate of the spatial field f_n (1 x h*n vector)
    DVector<double> g_curr_{};     // current PDE misfit
    DVector<double> beta_curr_{};  // current estimate of the coefficient vector (1 x h*q vector)
    DVector<double> f_prev_{};     // previous estimate of the spatial field f (1 x h*N vector)
    DVector<double> fn_prev_{};    // previous estimate of the spatial field f_n (1 x h*n vector)
    DVector<double> g_prev_{};     // previous PDE misfit
    DVector<double> beta_prev_{};  // previous estimate of the coefficient vector (1 x h*q vector)

    // room for algorithm quantities 
    SpMatrix<double> Ih_;                 // identity h x h 
    SpMatrix<double> In_;                 // identity n x n
    SpMatrix<double> Iq_;                 // identity q x q 
    SpMatrix<double> Ihn_;                // identity h*n x h*n 
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

    DVector<double> lambdas_;       // smoothing parameters in space   

    // constructor
    MSQRPDE() = default;
    MSQRPDE(const PDE& pde, std::vector<double>& alphas = {0.1, 0.5, 0.9}) : Base(pde), alphas_(alphas) {
      h_ = alphas_.size();
    }; 
    

    // ModelBase implementation
    void init_model() ;
    virtual void solve(); // finds a solution to the smoothing problem

    const bool crossing_constraints() const;

    void assemble_matrices(){

      std::cout << "MSQRPDE assemble: here 1" << std::endl;

      // room for solution 
      f_curr_.resize(h_*n_basis());
      fn_curr_.resize(h_*n_obs());
      g_curr_.resize(h_*n_basis());
      beta_curr_.resize(h_*q());
      f_prev_.resize(h_*n_basis());
      fn_prev_.resize(h_*n_obs());
      g_prev_.resize(h_*n_basis());
      beta_prev_.resize(h_*q());

      std::cout << "MSQRPDE assemble: here 2" << std::endl;

      // set all lambdas equal 
      lambdas_.resize(h_);   
      lambdas_ =  this->lambdaS()*DVector<double>::Ones(h_);  

      std::cout << "MSQRPDE assemble: here 3" << std::endl;

      // set identity matrices 
      Ih_.resize(h_, h_); 
	    Ih_.setIdentity();
      In_.resize(n_obs(), n_obs()); 
	    In_.setIdentity();
      Iq_.resize(n_obs(), n_obs()); 
	    Iq_.setIdentity();
      Ihn_.resize(h_*n_obs(), h_*n_obs()); 
	    Ihn_.setIdentity();

      std::cout << "MSQRPDE assemble: here 4" << std::endl;

      // assemble FEM, mass and stiffness matrices
      Psi_multiple_ = Kronecker(Ih_, Psi()); 
      std::cout << "MSQRPDE assemble: here 4.1" << std::endl;
      R0_multiple_ = Kronecker(SpMatrix<double>(DiagMatrix<double>(lambdas_)), R0());
      std::cout << "MSQRPDE assemble: here 4.5" << std::endl;
      R1_multiple_ = Kronecker(SpMatrix<double>(DiagMatrix<double>(lambdas_)), R1());

      std::cout << "MSQRPDE assemble: here 5" << std::endl;

      // assemble 
      DMatrix<double> Ih_dense(Ih_); 
      X_multiple_ = Kronecker(Ih_dense, X()); 

      std::cout << "MSQRPDE assemble: here 6" << std::endl;

      // BandMatrix<double> E_(h_-1, h_, 1, 1);
      // E_.diagonal(1).setConstant(1);
      // E_.diagonal(-1).setConstant(-1);
      SpMatrix<double> E_{};
      E_.resize(h_-1, h_); 
      E_.reserve(2*h_-1); 

      std::vector<fdaPDE::Triplet<double>> tripletListOnes;
      std::vector<fdaPDE::Triplet<double>> tripletListMinusOnes;
      tripletListMinusOnes.reserve(h_);
      tripletListOnes.reserve(h_-1);

      std::cout << "MSQRPDE assemble: here 7" << std::endl;

      for(std::size_t i = 0; i < h_; ++i)
	      tripletListMinusOnes.emplace_back(i, i, -1.0);

      for(std::size_t i = 0; i < h_-1; ++i)
	      tripletListOnes.emplace_back(i, i+1, 1.0);

      E_.setFromTriplets(tripletListOnes.begin(), tripletListOnes.end());
      E_.setFromTriplets(tripletListMinusOnes.begin(), tripletListMinusOnes.end());

      std::cout << "MSQRPDE assemble: here 8" << std::endl;

      D_ = Kronecker(E_, Iq_);        
      D_script_ = Kronecker(E_, In_);  

      std::cout << "MSQRPDE assemble: here 9" << std::endl;

    }

    DVector<double> rho_alpha(const double&, const DVector<double>&) const; 
    DVector<double> fitted(unsigned int j) const; 
    DVector<double> fitted() const; 
    double model_loss() const; 
    const std::pair<unsigned int, unsigned int> block_indexes(unsigned int, unsigned int) const; 

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