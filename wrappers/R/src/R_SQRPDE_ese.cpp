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
  
  void set_observations(const DMatrix<double>& data) {
    df_.template insert<double>(OBSERVATIONS_BLK, data);
  }
  void set_locations(const DMatrix<double>& data) {
    model_.set_spatial_locations(data);
  }

  void set_alpha(const double &alpha){
    model_.setAlpha(alpha); 
    return;
  }

  void setTolerances(double tol_w, double tol_fpirls){
    model_.setTolerances(tol_w, tol_fpirls); 
    return; 
  }

  void setMassLumpingGCV(bool lump){
    model_.setMassLumpingGCV(lump); 
    return; 
  }

  void setMassLumpingSystem(bool lump){
    regularization_.pde().setMassLumpingSystem(lump); 
    return; 
  }

  /* getters */
  SpMatrix<double> R0() const { return model_.R0(); }
  SpMatrix<double> Psi() { return model_.Psi(); }    // M tolto model_.Psi(not_nan());
    
  /* initialize model and solve smoothing problem */
  void solve() {
    model_.setData(df_);
    model_.init();
    model_.solve();
    return;
  }

  // metti quello che ti serve  
};



// definition of Rcpp module
typedef R_SQRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> SQRPDE_Laplacian_2D_GeoStatNodes;

// locations == nodes
RCPP_MODULE(SQRPDE_Laplacian_2D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_2D_GeoStatNodes>("SQRPDE_Laplacian_2D_GeoStatNodes")
    .constructor<Laplacian_2D_Order1>()
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_2D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_2D_GeoStatNodes::set_alpha)
    .method("setTolerances",     &SQRPDE_Laplacian_2D_GeoStatNodes::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_2D_GeoStatNodes::setMassLumpingGCV)
    .method("set_observations", &SQRPDE_Laplacian_2D_GeoStatNodes::set_observations)
    .method("R0",       &SQRPDE_Laplacian_2D_GeoStatNodes::R0)
    .method("init",       &SQRPDE_Laplacian_2D_GeoStatNodes::init)
    .method("init_regularization",       &SQRPDE_Laplacian_2D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2D_GeoStatNodes::init_pde)
    .method("Psi",       &SQRPDE_Laplacian_2D_GeoStatNodes::Psi)
    .method("solve",            &SQRPDE_Laplacian_2D_GeoStatNodes::solve);
}

// locations != nodes

typedef R_SQRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> SQRPDE_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(SQRPDE_Laplacian_2D_GeoStatLocations) {
  Rcpp::class_<SQRPDE_Laplacian_2D_GeoStatLocations>("SQRPDE_Laplacian_2D_GeoStatLocations")
    .constructor<Laplacian_2D_Order1>()
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_2D_GeoStatLocations::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_2D_GeoStatLocations::set_alpha)
    .method("setTolerances",     &SQRPDE_Laplacian_2D_GeoStatLocations::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_2D_GeoStatLocations::setMassLumpingGCV)
    .method("set_locations",  &SQRPDE_Laplacian_2D_GeoStatLocations::set_locations)
    .method("set_observations", &SQRPDE_Laplacian_2D_GeoStatLocations::set_observations)
    .method("R0",       &SQRPDE_Laplacian_2D_GeoStatLocations::R0)
    .method("init_regularization",       &SQRPDE_Laplacian_2D_GeoStatLocations::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_2D_GeoStatLocations::init_pde)
    .method("init",       &SQRPDE_Laplacian_2D_GeoStatLocations::init)
    .method("Psi",       &SQRPDE_Laplacian_2D_GeoStatLocations::Psi)
    .method("solve",            &SQRPDE_Laplacian_2D_GeoStatLocations::solve);
}

// 3D wrapper
typedef R_SQRPDE<Laplacian_3D_Order1, fdaPDE::models::GeoStatMeshNodes> SQRPDE_Laplacian_3D_GeoStatNodes;
RCPP_MODULE(SQRPDE_Laplacian_3D_GeoStatNodes) {
  Rcpp::class_<SQRPDE_Laplacian_3D_GeoStatNodes>("SQRPDE_Laplacian_3D_GeoStatNodes")
    .constructor<Laplacian_3D_Order1>()
    // setters
    .method("set_lambda_s",     &SQRPDE_Laplacian_3D_GeoStatNodes::set_lambda_s)
    .method("set_alpha",     &SQRPDE_Laplacian_3D_GeoStatNodes::set_alpha)
    .method("setTolerances",     &SQRPDE_Laplacian_3D_GeoStatNodes::setTolerances)
    .method("setMassLumpingGCV",     &SQRPDE_Laplacian_3D_GeoStatNodes::setMassLumpingGCV)
    .method("set_observations", &SQRPDE_Laplacian_3D_GeoStatNodes::set_observations)
    .method("init_regularization",       &SQRPDE_Laplacian_3D_GeoStatNodes::init_regularization)
    .method("init_pde",       &SQRPDE_Laplacian_3D_GeoStatNodes::init_pde)
    .method("R0",       &SQRPDE_Laplacian_3D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &SQRPDE_Laplacian_3D_GeoStatNodes::solve);
}
