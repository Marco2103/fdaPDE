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
RCPP_EXPOSED_AS  (Laplacian_25D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_25D_Order1)
RCPP_EXPOSED_AS  (Laplacian_1_5D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_1_5D_Order1)

// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (ConstantCoefficients_2D_Order1)
RCPP_EXPOSED_WRAP(ConstantCoefficients_2D_Order1)
// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (SpaceVarying_2D_Order1)
RCPP_EXPOSED_WRAP(SpaceVarying_2D_Order1)

// wrapper for SQRPDE module
template <typename RegularizingPDE, typename S> class R_SQRPDE{
protected:
  typedef RegularizingPDE RegularizingPDE_;
  RegularizingPDE_ regularization_; // tipo di PDE

  // qui metti il modello che vuoi wrappare 
  SQRPDE<typename RegularizingPDE_::PDEType, S> model_;
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
  void set_covariates(const DMatrix<double>& X) { df_.template insert<double>(DESIGN_MATRIX_BLK, X); }  // new
  void set_locations(const DMatrix<double>& locs) { model_.set_spatial_locations(locs); }
  void set_subdomains(const DMatrix<int>& areal) { model_.set_spatial_locations(areal); }
  void set_alpha(const double &alpha){ model_.setAlpha(alpha); }
  // void setTolerances(double tol_w, double tol_fpirls){ model_.setTolerances(tol_w, tol_fpirls); }
  void setMassLumpingGCV(bool lump){ model_.setMassLumpingGCV(lump); }
  void setMassLumpingSystem(bool lump){ regularization_.pde().setMassLumpingSystem(lump); }

  /* getters */
  SpMatrix<double> R0() const { return model_.R0(); }
  SpMatrix<double> Psi() { return model_.Psi(); }    // M tolto model_.Psi(not_nan());
  auto PsiTD() { return model_.PsiTD(); }   // aggiunto per areal
  DVector<double> f() const { return model_.f(); }; 
  DVector<double> fn() const { return model_.Psi()*model_.f(); }; 
  DVector<double> beta() const { return model_.beta(); }; 

  // getters aggiunti per areal
  // auto PsiTD(not_nan) const { return Psi_.transpose()*D_; }
  // std::size_t n_spatial_locs() const { return subdomains_.rows(); }
  // const DiagMatrix<double>& D() const { return model_.D(); } // prima decommentato
  const DMatrix<int>& locs() const { return model_.locs(); } // prima decommentato
    
  /* initialize model and solve smoothing problem */
  void solve() {
    model_.setData(df_);
    model_.init();
    model_.solve();
    return;
  }

};



// definition of Rcpp modules

// locations == nodes

  // laplacian 
typedef R_SQRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> SQRPDE_Laplacian_2D_GeoStatNodes;
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
    // .method("setTolerances",     &SQRPDE_Laplacian_2D_GeoStatNodes::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_2D_GeoStatNodes::setMassLumpingGCV)
    .method("set_observations", &SQRPDE_Laplacian_2D_GeoStatNodes::set_observations)
    .method("init",       &SQRPDE_Laplacian_2D_GeoStatNodes::init)
    .method("init_regularization",       &SQRPDE_Laplacian_2D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2D_GeoStatNodes::init_pde)
    .method("solve",            &SQRPDE_Laplacian_2D_GeoStatNodes::solve);
}

  // constant PDE coefficients 
typedef R_SQRPDE<ConstantCoefficients_2D_Order1, fdaPDE::models::GeoStatMeshNodes> SQRPDE_ConstantCoefficients_2D_GeoStatNodes;
RCPP_MODULE(SQRPDE_ConstantCoefficients_2D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_ConstantCoefficients_2D_GeoStatNodes>("SQRPDE_ConstantCoefficients_2D_GeoStatNodes")
    .constructor<ConstantCoefficients_2D_Order1>()
    //getters 
    .method("f",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::f)
    .method("fn",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::fn)
    .method("beta",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::beta)
    .method("R0",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::R0)
    .method("Psi",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::Psi)
    // setters
    .method("set_lambda_s",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::set_alpha)
    // .method("setTolerances",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::setMassLumpingGCV)
    .method("set_observations", &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::set_observations)
    .method("init",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::init)
    .method("init_regularization",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::init_pde)
    .method("solve",            &SQRPDE_ConstantCoefficients_2D_GeoStatNodes::solve);
}

// locations != nodes
typedef R_SQRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> SQRPDE_Laplacian_2D_GeoStatLocations;
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
    // .method("setTolerances",     &SQRPDE_Laplacian_2D_GeoStatLocations::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_2D_GeoStatLocations::setMassLumpingGCV)
    .method("set_locations",  &SQRPDE_Laplacian_2D_GeoStatLocations::set_locations)
    .method("set_observations", &SQRPDE_Laplacian_2D_GeoStatLocations::set_observations)
    .method("init",       &SQRPDE_Laplacian_2D_GeoStatLocations::init)
    .method("init_regularization",       &SQRPDE_Laplacian_2D_GeoStatLocations::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2D_GeoStatLocations::init_pde)
    .method("solve",            &SQRPDE_Laplacian_2D_GeoStatLocations::solve);
}

// areal
typedef R_SQRPDE<Laplacian_2D_Order1, fdaPDE::models::Areal> SQRPDE_Laplacian_2D_Areal;   // era sbagliato
RCPP_MODULE(SQRPDE_Laplacian_2D_Areal) {
  Rcpp::class_<SQRPDE_Laplacian_2D_Areal>("SQRPDE_Laplacian_2D_Areal")
    .constructor<Laplacian_2D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_2D_Areal::f)
    .method("fn",     &SQRPDE_Laplacian_2D_Areal::fn)
    .method("beta",     &SQRPDE_Laplacian_2D_Areal::beta)
    .method("R0",       &SQRPDE_Laplacian_2D_Areal::R0)
    .method("Psi",       &SQRPDE_Laplacian_2D_Areal::Psi)
    .method("locs",       &SQRPDE_Laplacian_2D_Areal::locs)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_2D_Areal::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_2D_Areal::set_alpha)
    // .method("setTolerances",     &SQRPDE_Laplacian_2D_Areal::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_2D_Areal::setMassLumpingGCV)
    .method("set_subdomains", &SQRPDE_Laplacian_2D_Areal::set_subdomains)
    .method("set_observations", &SQRPDE_Laplacian_2D_Areal::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_2D_Areal::set_covariates)
    .method("init",       &SQRPDE_Laplacian_2D_Areal::init)
    .method("init_regularization",       &SQRPDE_Laplacian_2D_Areal::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2D_Areal::init_pde)
    .method("R0",       &SQRPDE_Laplacian_2D_Areal::R0)
    .method("solve",            &SQRPDE_Laplacian_2D_Areal::solve);
}


// 3D wrappers 

// locations = nodes 
typedef R_SQRPDE<Laplacian_3D_Order1, fdaPDE::models::GeoStatMeshNodes> SQRPDE_Laplacian_3D_GeoStatNodes;
RCPP_MODULE(SQRPDE_Laplacian_3D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_3D_GeoStatNodes>("SQRPDE_Laplacian_3D_GeoStatNodes")
    .constructor<Laplacian_3D_Order1>()
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_3D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_3D_GeoStatNodes::set_alpha)
    // .method("setTolerances",     &SQRPDE_Laplacian_3D_GeoStatNodes::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_3D_GeoStatNodes::setMassLumpingGCV)
    .method("set_observations", &SQRPDE_Laplacian_3D_GeoStatNodes::set_observations)
    .method("init_regularization",       &SQRPDE_Laplacian_3D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_3D_GeoStatNodes::init_pde)
    .method("R0",       &SQRPDE_Laplacian_3D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_3D_GeoStatNodes::solve);
}

// 3D, locations != nodes, laplacian 
typedef R_SQRPDE<Laplacian_3D_Order1, fdaPDE::models::GeoStatMeshNodes> SQRPDE_Laplacian_3D_GeoStatLocations;
RCPP_MODULE(SQRPDE_Laplacian_3D_GeoStatLocations) {
  Rcpp::class_<SQRPDE_Laplacian_3D_GeoStatLocations>("SQRPDE_Laplacian_3D_GeoStatLocations")
    .constructor<Laplacian_3D_Order1>()
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_3D_GeoStatLocations::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_3D_GeoStatLocations::set_alpha)
    // .method("setTolerances",     &SQRPDE_Laplacian_3D_GeoStatLocations::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_3D_GeoStatLocations::setMassLumpingGCV)
    .method("set_locations",  &SQRPDE_Laplacian_3D_GeoStatLocations::set_locations)
    .method("set_observations", &SQRPDE_Laplacian_3D_GeoStatLocations::set_observations)
    .method("init_regularization",       &SQRPDE_Laplacian_3D_GeoStatLocations::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_3D_GeoStatLocations::init_pde)
    .method("R0",       &SQRPDE_Laplacian_3D_GeoStatLocations::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_3D_GeoStatLocations::solve);
}


// 2.5D wrappers 

// locations = nodes 
typedef R_SQRPDE<Laplacian_25D_Order1, fdaPDE::models::GeoStatMeshNodes> SQRPDE_Laplacian_25D_GeoStatNodes;
RCPP_MODULE(SQRPDE_Laplacian_25D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_25D_GeoStatNodes>("SQRPDE_Laplacian_25D_GeoStatNodes")
    .constructor<Laplacian_25D_Order1>()
    //getters 
    .method("f",     &SQRPDE_Laplacian_25D_GeoStatNodes::f)
    .method("fn",     &SQRPDE_Laplacian_25D_GeoStatNodes::fn)
    .method("beta",     &SQRPDE_Laplacian_25D_GeoStatNodes::beta)
    .method("R0",       &SQRPDE_Laplacian_25D_GeoStatNodes::R0)
    .method("Psi",       &SQRPDE_Laplacian_25D_GeoStatNodes::Psi)
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_25D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_25D_GeoStatNodes::set_alpha)
    // .method("setTolerances",     &SQRPDE_Laplacian_25D_GeoStatNodes::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_25D_GeoStatNodes::setMassLumpingGCV)
    .method("set_observations", &SQRPDE_Laplacian_25D_GeoStatNodes::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_25D_GeoStatNodes::set_covariates)
    .method("init_regularization",       &SQRPDE_Laplacian_25D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_25D_GeoStatNodes::init_pde)
    .method("R0",       &SQRPDE_Laplacian_25D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_25D_GeoStatNodes::solve);
}


// 1.5D wrappers 

// locations = nodes 
typedef R_SQRPDE<Laplacian_1_5D_Order1, fdaPDE::models::GeoStatMeshNodes> SQRPDE_Laplacian_1_5D_GeoStatNodes;
RCPP_MODULE(SQRPDE_Laplacian_1_5D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_1_5D_GeoStatNodes>("SQRPDE_Laplacian_25D_GeoStatNodes")
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
    // .method("setTolerances",     &SQRPDE_Laplacian_1_5D_GeoStatNodes::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_1_5D_GeoStatNodes::setMassLumpingGCV)
    .method("set_observations", &SQRPDE_Laplacian_1_5D_GeoStatNodes::set_observations)
    .method("set_covariates", &SQRPDE_Laplacian_1_5D_GeoStatNodes::set_covariates)
    .method("init_regularization",       &SQRPDE_Laplacian_1_5D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_1_5D_GeoStatNodes::init_pde)
    .method("R0",       &SQRPDE_Laplacian_1_5D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_1_5D_GeoStatNodes::solve);
}

