// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <thread>
#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/regression/SQRPDE.h>
using fdaPDE::models::SQRPDE;
#include <fdaPDE/core/utils/DataStructures/BlockFrame.h>
#include <fdaPDE/models/ModelTraits.h>
#include <fdaPDE/core/FEM/PDE.h>
using fdaPDE::core::FEM::DefaultOperator;
using fdaPDE::core::FEM::PDE;
#include <fdaPDE/core/MESH/Mesh.h>
using fdaPDE::core::MESH::Mesh;
#include <fdaPDE/models/SamplingDesign.h>
#include <fdaPDE/core/MESH/engines/AlternatingDigitalTree/ADT.h>
using fdaPDE::core::MESH::ADT;

#include "Common.h"

RCPP_EXPOSED_AS  (Laplacian_2D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_2D_Order1)
RCPP_EXPOSED_AS  (Laplacian_3D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_3D_Order1)
RCPP_EXPOSED_AS  (Laplacian_2_5D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_2_5D_Order1)
RCPP_EXPOSED_AS  (Laplacian_1_5D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_1_5D_Order1)

// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (ConstantCoefficients_2D_Order1)
RCPP_EXPOSED_WRAP(ConstantCoefficients_2D_Order1)
// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (SpaceVarying_2D_Order1)
RCPP_EXPOSED_WRAP(SpaceVarying_2D_Order1)

// wrapper for SQRPDE module
template <typename RegularizingPDE, typename RegularizationType, typename S, typename Solver> class R_SQRPDE{
protected:
  typedef RegularizingPDE RegularizingPDE_;
  RegularizingPDE_ regularization_; // tipo di PDE

  // qui metti il modello che vuoi wrappare 
  SQRPDE<typename RegularizingPDE_::PDEType, RegularizationType, S, Solver> model_;
  BlockFrame<double, int> df_;
public:
  
  R_SQRPDE(const RegularizingPDE_ &regularization)
    : regularization_(regularization){
    model_.setPDE(regularization_.pde());
  };

  // metodi di init,  messi a caso giusto nel caso dovessero servire

  void init_pde() { model_.init_pde(); }
  void init() { model_.init(); }
  void init_regularization() { model_.init_pde(); model_.init_regularization(); model_.init_sampling(); }
  
  /* setters */
  void set_lambda_s(double lambdaS) { model_.setLambdaS(lambdaS); }
  void set_observations(const DMatrix<double>& data) { df_.template insert<double>(OBSERVATIONS_BLK, data); }
  void set_covariates(const DMatrix<double>& X) { df_.template insert<double>(DESIGN_MATRIX_BLK, X); }  
  void set_locations(const DMatrix<double>& locs) { model_.set_spatial_locations(locs); }
  void set_subdomains(const DMatrix<int>& areal) { model_.set_spatial_locations(areal); }
  void set_alpha(const double &alpha){ model_.setAlpha(alpha); }
  
  /* getters */
  SpMatrix<double> R0() const { return model_.R0(); }
  SpMatrix<double> R1() const { return model_.R1(); }
  SpMatrix<double> Psi() { return model_.Psi(); }  
  DMatrix<double> u() { return model_.u(); }  
  auto PsiTD() { return model_.PsiTD(); }   
  DVector<double> f() const { return model_.f(); }; 
  DVector<double> fn() const { return model_.Psi()*model_.f(); }; 
  DVector<double> beta() const { return model_.beta(); }; 
  DMatrix<double> locs() const { return model_.locs(); }; 
  DMatrix<int> subdomains() const { return model_.locs(); };  

    
  /* initialize model and solve smoothing problem */
  void solve() {
    model_.setData(df_);
    model_.init();
    model_.solve();
    return;
  }

};

// definition of Rcpp modules

// 2D, locations == nodes, laplacian 
typedef R_SQRPDE<Laplacian_2D_Order1, fdaPDE::models::SpaceOnly, 
                fdaPDE::models::GeoStatMeshNodes, fdaPDE::models::MonolithicSolver> SQRPDE_Laplacian_2D_GeoStatNodes;
RCPP_MODULE(SQRPDE_Laplacian_2D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_2D_GeoStatNodes>("SQRPDE_Laplacian_2D_GeoStatNodes")
    .constructor<Laplacian_2D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_2D_GeoStatNodes::f)
    .method("fn",     &SQRPDE_Laplacian_2D_GeoStatNodes::fn)
    .method("beta",     &SQRPDE_Laplacian_2D_GeoStatNodes::beta)
    .method("R0",       &SQRPDE_Laplacian_2D_GeoStatNodes::R0)
    .method("Psi",       &SQRPDE_Laplacian_2D_GeoStatNodes::Psi)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_2D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_2D_GeoStatNodes::set_alpha)
    .method("set_observations", &SQRPDE_Laplacian_2D_GeoStatNodes::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_2D_GeoStatNodes::set_covariates)
    .method("init",       &SQRPDE_Laplacian_2D_GeoStatNodes::init)
    .method("init_regularization",       &SQRPDE_Laplacian_2D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2D_GeoStatNodes::init_pde)
    .method("solve",            &SQRPDE_Laplacian_2D_GeoStatNodes::solve);
}

// 2D, locations == nodes, constant PDE coefficients 
typedef R_SQRPDE<ConstantCoefficients_2D_Order1,
                fdaPDE::models::SpaceOnly, fdaPDE::models::GeoStatMeshNodes, 
                fdaPDE::models::MonolithicSolver> SQRPDE_ConstantCoefficients_2D_GeoStatNodes;
RCPP_MODULE(SQRPDE_ConstantCoefficients_2D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_ConstantCoefficients_2D_GeoStatNodes>("SQRPDE_ConstantCoefficients_2D_GeoStatNodes")
    .constructor<ConstantCoefficients_2D_Order1>()
    //getters 
    .method("f",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::f)
    .method("fn",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::fn)
    .method("beta",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::beta)
    .method("R0",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::R0)
    .method("R1",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::R1)
    .method("Psi",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::Psi)
    .method("u",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::u)
    // setters
    .method("set_lambda_s",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::set_alpha)
    .method("set_observations", &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::set_observations)
    .method("set_covariates", &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::set_covariates)
    .method("init",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::init)
    .method("init_regularization",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::init_pde)
    .method("solve",            &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::solve);
}

// 2D, locations != nodes, laplacian 
typedef R_SQRPDE<Laplacian_2D_Order1, fdaPDE::models::SpaceOnly, 
                fdaPDE::models::GeoStatLocations, fdaPDE::models::MonolithicSolver> SQRPDE_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(SQRPDE_Laplacian_2D_GeoStatLocations) {
  Rcpp::class_<SQRPDE_Laplacian_2D_GeoStatLocations>("SQRPDE_Laplacian_2D_GeoStatLocations")
    .constructor<Laplacian_2D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_2D_GeoStatLocations::f)
    .method("fn",     &SQRPDE_Laplacian_2D_GeoStatLocations::fn)
    .method("beta",     &SQRPDE_Laplacian_2D_GeoStatLocations::beta)
    .method("R0",       &SQRPDE_Laplacian_2D_GeoStatLocations::R0)
    .method("Psi",       &SQRPDE_Laplacian_2D_GeoStatLocations::Psi)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_2D_GeoStatLocations::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_2D_GeoStatLocations::set_alpha)
    .method("set_locations",  &SQRPDE_Laplacian_2D_GeoStatLocations::set_locations)
    .method("set_observations", &SQRPDE_Laplacian_2D_GeoStatLocations::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_2D_GeoStatLocations::set_covariates)
    .method("init",       &SQRPDE_Laplacian_2D_GeoStatLocations::init)
    .method("init_regularization",       &SQRPDE_Laplacian_2D_GeoStatLocations::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2D_GeoStatLocations::init_pde)
    .method("solve",            &SQRPDE_Laplacian_2D_GeoStatLocations::solve);
}

// // 2D, locations != nodes, SpaceVarying
// typedef R_SQRPDE<SpaceVarying_2D_Order1, fdaPDE::models::SpaceOnly, 
//                 fdaPDE::models::GeoStatLocations, fdaPDE::models::MonolithicSolver> SQRPDE_SpaceVarying_2D_GeoStatLocations;
// RCPP_MODULE(SQRPDE_SpaceVarying_2D_GeoStatLocations) {
//   Rcpp::class_<SQRPDE_SpaceVarying_2D_GeoStatLocations>("SQRPDE_SpaceVarying_2D_GeoStatLocations")
//     .constructor<SpaceVarying_2D_Order1>()
//     //getters 
//     .method("f",     &SQRPDE_SpaceVarying_2D_GeoStatLocations::f)
//     .method("fn",     &SQRPDE_SpaceVarying_2D_GeoStatLocations::fn)
//     .method("beta",     &SQRPDE_SpaceVarying_2D_GeoStatLocations::beta)
//     .method("R0",       &SQRPDE_SpaceVarying_2D_GeoStatLocations::R0)
//     .method("Psi",       &SQRPDE_SpaceVarying_2D_GeoStatLocations::Psi)
//     // setters
//     .method("set_lambda_s",     &SQRPDE_SpaceVarying_2D_GeoStatLocations::set_lambda_s)
//     .method("set_alpha",     &SQRPDE_SpaceVarying_2D_GeoStatLocations::set_alpha)
//     .method("set_locations",  &SQRPDE_SpaceVarying_2D_GeoStatLocations::set_locations)
//     .method("set_observations", &SQRPDE_SpaceVarying_2D_GeoStatLocations::set_observations)
//     .method("set_covariates", &SQRPDE_SpaceVarying_2D_GeoStatLocations::set_covariates)
//     .method("init",       &SQRPDE_SpaceVarying_2D_GeoStatLocations::init)
//     .method("init_regularization",       &SQRPDE_SpaceVarying_2D_GeoStatLocations::init_regularization)
//     .method("init_pde",       &SQRPDE_SpaceVarying_2D_GeoStatLocations::init_pde)
//     .method("solve",            &SQRPDE_SpaceVarying_2D_GeoStatLocations::solve);
// }

// 2D, areal, laplacian 
typedef R_SQRPDE<Laplacian_2D_Order1, fdaPDE::models::SpaceOnly, 
                fdaPDE::models::Areal, fdaPDE::models::MonolithicSolver> SQRPDE_Laplacian_2D_Areal;   // era sbagliato
RCPP_MODULE(SQRPDE_Laplacian_2D_Areal) {
  Rcpp::class_<SQRPDE_Laplacian_2D_Areal>("SQRPDE_Laplacian_2D_Areal")
    .constructor<Laplacian_2D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_2D_Areal::f)
    .method("fn",     &SQRPDE_Laplacian_2D_Areal::fn)
    .method("beta",     &SQRPDE_Laplacian_2D_Areal::beta)
    .method("R0",       &SQRPDE_Laplacian_2D_Areal::R0)
    .method("Psi",       &SQRPDE_Laplacian_2D_Areal::Psi)
    .method("subdomains",       &SQRPDE_Laplacian_2D_Areal::subdomains)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_2D_Areal::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_2D_Areal::set_alpha)
    .method("set_subdomains", &SQRPDE_Laplacian_2D_Areal::set_subdomains)
    .method("set_observations", &SQRPDE_Laplacian_2D_Areal::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_2D_Areal::set_covariates)
    .method("init",       &SQRPDE_Laplacian_2D_Areal::init)
    .method("init_regularization",       &SQRPDE_Laplacian_2D_Areal::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2D_Areal::init_pde)
    .method("R0",       &SQRPDE_Laplacian_2D_Areal::R0)
    .method("solve",            &SQRPDE_Laplacian_2D_Areal::solve);
}


// 3D, locations == nodes, laplacian 
typedef R_SQRPDE<Laplacian_3D_Order1, fdaPDE::models::SpaceOnly,
                fdaPDE::models::GeoStatMeshNodes, fdaPDE::models::MonolithicSolver> SQRPDE_Laplacian_3D_GeoStatNodes;
RCPP_MODULE(SQRPDE_Laplacian_3D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_3D_GeoStatNodes>("SQRPDE_Laplacian_3D_GeoStatNodes")
    .constructor<Laplacian_3D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_3D_GeoStatNodes::f)
    .method("fn",     &SQRPDE_Laplacian_3D_GeoStatNodes::fn)
    .method("beta",     &SQRPDE_Laplacian_3D_GeoStatNodes::beta)
    .method("R0",       &SQRPDE_Laplacian_3D_GeoStatNodes::R0)
    .method("Psi",       &SQRPDE_Laplacian_3D_GeoStatNodes::Psi)
    .method("locs",       &SQRPDE_Laplacian_3D_GeoStatNodes::locs)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_3D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_3D_GeoStatNodes::set_alpha)
    .method("set_observations", &SQRPDE_Laplacian_3D_GeoStatNodes::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_3D_GeoStatNodes::set_covariates)
    .method("init_regularization",       &SQRPDE_Laplacian_3D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_3D_GeoStatNodes::init_pde)
    .method("R0",       &SQRPDE_Laplacian_3D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_3D_GeoStatNodes::solve);
}

// 3D, locations != nodes, laplacian 
typedef R_SQRPDE<Laplacian_3D_Order1, fdaPDE::models::SpaceOnly,
                fdaPDE::models::GeoStatLocations, fdaPDE::models::MonolithicSolver> SQRPDE_Laplacian_3D_GeoStatLocations;
RCPP_MODULE(SQRPDE_Laplacian_3D_GeoStatLocations) {
  Rcpp::class_<SQRPDE_Laplacian_3D_GeoStatLocations>("SQRPDE_Laplacian_3D_GeoStatLocations")
    .constructor<Laplacian_3D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_3D_GeoStatLocations::f)
    .method("fn",     &SQRPDE_Laplacian_3D_GeoStatLocations::fn)
    .method("beta",     &SQRPDE_Laplacian_3D_GeoStatLocations::beta)
    .method("R0",       &SQRPDE_Laplacian_3D_GeoStatLocations::R0)
    .method("Psi",       &SQRPDE_Laplacian_3D_GeoStatLocations::Psi)
    .method("locs",       &SQRPDE_Laplacian_3D_GeoStatLocations::locs)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_3D_GeoStatLocations::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_3D_GeoStatLocations::set_alpha)
    .method("set_locations",  &SQRPDE_Laplacian_3D_GeoStatLocations::set_locations)
    .method("set_observations", &SQRPDE_Laplacian_3D_GeoStatLocations::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_3D_GeoStatLocations::set_covariates)
    .method("init_regularization",       &SQRPDE_Laplacian_3D_GeoStatLocations::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_3D_GeoStatLocations::init_pde)
    .method("R0",       &SQRPDE_Laplacian_3D_GeoStatLocations::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_3D_GeoStatLocations::solve);
}

// 2.5, locations = nodes , laplacian
typedef R_SQRPDE<Laplacian_2_5D_Order1, fdaPDE::models::SpaceOnly,
                fdaPDE::models::GeoStatMeshNodes, fdaPDE::models::MonolithicSolver> SQRPDE_Laplacian_2_5D_GeoStatNodes;
RCPP_MODULE(SQRPDE_Laplacian_2_5D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_2_5D_GeoStatNodes>("SQRPDE_Laplacian_2_5D_GeoStatNodes")
    .constructor<Laplacian_2_5D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_2_5D_GeoStatNodes::f)
    .method("fn",     &SQRPDE_Laplacian_2_5D_GeoStatNodes::fn)
    .method("beta",     &SQRPDE_Laplacian_2_5D_GeoStatNodes::beta)
    .method("R0",       &SQRPDE_Laplacian_2_5D_GeoStatNodes::R0)
    .method("Psi",       &SQRPDE_Laplacian_2_5D_GeoStatNodes::Psi)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_2_5D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_2_5D_GeoStatNodes::set_alpha)
    .method("set_observations", &SQRPDE_Laplacian_2_5D_GeoStatNodes::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_2_5D_GeoStatNodes::set_covariates)
    .method("init_regularization",       &SQRPDE_Laplacian_2_5D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2_5D_GeoStatNodes::init_pde)
    .method("R0",       &SQRPDE_Laplacian_2_5D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_2_5D_GeoStatNodes::solve);
}


// 1.5, locations = nodes , laplacian
typedef R_SQRPDE<Laplacian_1_5D_Order1, fdaPDE::models::SpaceOnly, 
                fdaPDE::models::GeoStatMeshNodes, fdaPDE::models::MonolithicSolver> SQRPDE_Laplacian_1_5D_GeoStatNodes;
RCPP_MODULE(SQRPDE_Laplacian_1_5D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_1_5D_GeoStatNodes>("SQRPDE_Laplacian_1_5D_GeoStatNodes")
    .constructor<Laplacian_1_5D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_1_5D_GeoStatNodes::f)
    .method("fn",     &SQRPDE_Laplacian_1_5D_GeoStatNodes::fn)
    .method("beta",     &SQRPDE_Laplacian_1_5D_GeoStatNodes::beta)
    .method("R0",       &SQRPDE_Laplacian_1_5D_GeoStatNodes::R0)
    .method("Psi",       &SQRPDE_Laplacian_1_5D_GeoStatNodes::Psi)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_1_5D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_1_5D_GeoStatNodes::set_alpha)
    .method("set_observations", &SQRPDE_Laplacian_1_5D_GeoStatNodes::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_1_5D_GeoStatNodes::set_covariates)
    .method("init_regularization",       &SQRPDE_Laplacian_1_5D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_1_5D_GeoStatNodes::init_pde)
    .method("R0",       &SQRPDE_Laplacian_1_5D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_1_5D_GeoStatNodes::solve);
}

