// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/regression/STRPDE.h>
using fdaPDE::models::STRPDE;
#include <fdaPDE/core/utils/DataStructures/BlockFrame.h>
#include <fdaPDE/models/ModelTraits.h>
#include <fdaPDE/core/FEM/PDE.h>
using fdaPDE::core::FEM::DefaultOperator;
using fdaPDE::core::FEM::PDE;
#include <fdaPDE/core/FEM/EigenValueProblem.h>
using fdaPDE::core::FEM::EigenValueProblem;
#include <fdaPDE/core/MESH/Mesh.h>
using fdaPDE::core::MESH::Mesh;
#include <fdaPDE/models/SamplingDesign.h>
#include <fdaPDE/core/FEM/Evaluator.h>
using fdaPDE::core::FEM::Evaluator;
#include <fdaPDE/calibration/GCV.h>
#include <fdaPDE/calibration/StochasticEDF.h>
using fdaPDE::calibration::GCV;
using fdaPDE::calibration::StochasticEDF;
#include <fdaPDE/core/OPT/optimizers/GridOptimizer.h>
using fdaPDE::core::OPT::GridOptimizer;

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

// wrapper for SRPDE module
template <typename RegularizingPDE, typename S> class R_STRPDE {
protected:
  typedef RegularizingPDE RegularizingPDE_;
  RegularizingPDE_ regularization_;
  /* the model this Rcpp module wraps */
  typedef STRPDE<typename RegularizingPDE_::PDEType, fdaPDE::models::SpaceTimeSeparable, S, fdaPDE::models::MonolithicSolver> ModelType;
  ModelType model_;
  BlockFrame<double, int> df_;

  //GCV<ModelType, StochasticEDF<ModelType>> gcv_ptr;

  StochasticEDF<ModelType> trS_compute;
  double trS_;

  bool inited_ = false;
  
public:
  R_STRPDE(const RegularizingPDE_ &regularization)
    : regularization_(regularization), trS_compute(model_, 100) {
    model_.setPDE(regularization_.pde());
  };

  /* setters */
  void set_lambda_s(double lambdaS) { model_.setLambdaS(lambdaS); inited_ = false; }
  void set_lambda_t(double lambdaT) { model_.setLambdaT(lambdaT); inited_ = false; }
  void set_time_domain(const DVector<double>& t) {
    model_.setTimeDomain(t);
    inited_ = false;
  }
  void set_time_locations(const DMatrix<double>& time_locs) {
    model_.set_temporal_locations(time_locs);
    inited_ = false;
  }
  void set_observations(const DMatrix<double>& data) {
    df_.template insert<double>(OBSERVATIONS_BLK, data);
  }
  void set_locations(const DMatrix<double>& data) {
    model_.set_spatial_locations(data);
    inited_ = false;
  }

  /* getters */
  DMatrix<double> fitted() const { return model_.fitted(); }

  // SpMatrix<double> R0() const { return model_.R0(); }
  // void init() { model_.setData(df_); model_.init_pde(); model_.init_regularization(); }
  // SpMatrix<double> Psi() {
  //   model_.setData(df_);
  //   model_.init_sampling();
  //   return model_.Psi();
  // }

  void init() {
    model_.setData(df_);
    model_.init();
  }

  void update_missing() {
    model_.setData(df_);
    model_.init_sampling(true);
    model_.init_nan();
  }
  
  SpMatrix<double> Psi() const { return model_.Psi(); }
  SpMatrix<double> R0() const { return model_.R0(); }

  SpMatrix<double> mass() const { return model_.pde().R0(); }
  
  /* initialize model and solve smoothing problem, finds model with optimal \lambda according to GCV */
  double solve() {
    model_.setData(df_);
    if(!inited_){
      model_.init();
      trS_ = trS_compute.compute();
      inited_ = true;
      std::cout << "mi inizializzo" << std::endl;
    }

    // // define GCV calibrator
    // GCV<decltype(model_), StochasticEDF<decltype(model_)>> GCV(model_, 100);
    // GridOptimizer<2> opt;

    // std::vector<SVector<2>> lambdas;
    // for(int i = 0; i < lambdaS.size(); ++i) {
    //   for(int j = 0; j < lambdaT.size(); ++j){
    // 	lambdas.push_back(SVector<2>(lambdaS[i], lambdaT[j]));
    //   }
    // }
    // ScalarField<2, GCV<ModelType, StochasticEDF<ModelType>>> obj(*gcv_ptr);
    // opt.optimize(obj, lambdas); // optimize gcv field
    // SVector<2> best_lambda = opt.optimum();

    // std::cout << "optimal lambda" << std::endl;
    // std::cout << best_lambda << std::endl;
    
    //model_.setLambda(best_lambda);
    //model_.init();
    
    model_.solve();

    double dor = model_.n_obs() - (model_.q() + trS_); // (n - (q + Tr[S])
    return (model_.n_obs()/std::pow(dor, 2))*( model_.norm(model_.fitted(), model_.y())); 
    
    //return gcv_ptr.eval();
    
    // std::vector<SVector<2>> lambdas;
    // lambdas.clear();
    // for(double x = -4.0; x <= -2.0; x += 0.5) {
    //   for(double y = -4.0; y <= -2.0; y += 0.5) {
    // 	lambdas.push_back(SVector<2>(std::pow(10,x), std::pow(10,y)));
    //   }
    // }
    // model_.setLambda(lambdas);
  }
  
  // Rcpp::List plot_basis(std::size_t i) {
  //   regularization_.pde().init();
  //   //regularization_.pde().solve();
    
  //   //Eigen::GeneralizedSelfAdjointEigenSolver<DMatrix<double>> ges;
  //   // compute generalized eigendecomposition of matrices R_1 and R_0:
  //   // find k \in \mathbb{R} and vector f such that R_1*f = k*R_0*f
  //   //ges.setMaxIterations(16);
  //   //ges.compute(DMatrix<double>(regularization_.pde().R1()), DMatrix<double>(regularization_.pde().R0()));

  //   EigenValueProblem<decltype(regularization_.pde())> evp(regularization_.pde());
  //   evp.solve();
    
  //   //std::cout << ges.eigenvalues() << std::endl;
    
  //   // define eigen solver
  //   // Eigen::GeneralizedEigenSolver<DMatrix<double>> ges;
  //   // // compute generalized eigendecomposition of matrices R_1 and R_0:
  //   // // find k \in \mathbb{R} and vector f such that R_1*f = k*R_0*f
  //   // ges.setMaxIterations(16);
  //   // ges.compute(DMatrix<double>(regularization_.pde().R1()), DMatrix<double>(regularization_.pde().R0()));

  //   // std::size_t idx = 0;
  //   // double min_eigen = ges.eigenvalues()[0];
  //   // for(std::size_t i = 1; i < ges.eigenvalues().rows(); ++i){
  //   //   if(ges.eigenvalues()[i] < min_eigen){
  //   // 	idx = i;
  //   // 	min_eigen = ges.eigenvalues()[idx];
  //   //   }	
  //   // }
    
  //   // DVector<double> result = ges.eigenvectors().col(i);
  //   DVector<double> result = evp.eigenfunctions().col(i);
  //   // result.resize(ges.eigenvectors().col(idx).rows());
  //   // for(std::size_t j = 0; j < ges.eigenvectors().col(idx).rows(); ++j)
  //   //   result[j] = ges.eigenvectors().col(idx).coeff(j,0);

  //   result = result;
    
  //   // LagrangianBasis<2,2,1> basis{};
  //   // double resolution = 0.01;
  //   // std::size_t NN = 1/resolution;
  //   // // print stuffs required by R
  //   // std::vector<double> x_coord{};
  //   // x_coord.resize(NN*NN);
  //   // std::vector<double> y_coord{};
  //   // y_coord.resize(NN*NN);
  //   // std::vector<double> data{};
  //   // data.resize(NN*NN);

  //   // // prepare for R layer
  //   // std::size_t j = 0;
  //   // std::size_t z = 0;
  //   // for(double x = 0; x < 1; x+=resolution, ++j){
  //   //   for(double y = 0; y < 1; y+=resolution, ++z){
  //   // 	x_coord[j*NN + z] = x;
  //   // 	y_coord[j*NN + z] = y;
  //   // 	if(1-x > y){
  //   // 	  data[j*NN + z] = basis[i](SVector<2>(x,y));
  //   // 	}else
  //   // 	  data[j*NN + z] = std::numeric_limits<double>::quiet_NaN();
  //   //   }
  //   //   z = 0;
  //   // }
  //   // return Rcpp::List::create(Rcpp::Named("x") = x_coord, Rcpp::Named("y") = y_coord, Rcpp::Named("solution") = data);

  //   return Rcpp::List::create(Rcpp::Named("eigenfunc") = evp.eigenfunctions(), Rcpp::Named("eigenval") = evp.eigenvalues());
  // }
  
  // Rcpp::List plot(double resolution, DMatrix<double> solution) const {
  //   // solution evaluation
  //   Evaluator<std::decay<decltype(regularization_.pde())>::type::local_dimension,
  // 	      std::decay<decltype(regularization_.pde())>::type::embedding_dimension,
  // 	      std::decay<decltype(regularization_.pde())>::type::basis_order> eval;
  //   fdaPDE::core::FEM::Raster<std::decay<decltype(regularization_.pde())>::type::local_dimension> img
  //     = eval.toRaster(model_.pde().domain(), solution, resolution, model_.pde().basis());
  //   // print stuffs required by R
  //   std::vector<double> x_coord{};
  //   x_coord.resize(img.size());
  //   std::vector<double> y_coord{};
  //   y_coord.resize(img.size());
  //   // prepare for R layer
  //   for(std::size_t i = 0; i < img.size(); ++i){
  //     x_coord[i] = img.coords_[i][0];
  //     y_coord[i] = img.coords_[i][1];
  //   }

  //   return Rcpp::List::create(Rcpp::Named("x") = x_coord,
  // 			      Rcpp::Named("y") = y_coord,
  // 			      Rcpp::Named("solution") = img.data_);
  // }

  DVector<double> eval_to_locs(const DMatrix<double>& coeff, const DMatrix<double>& locs) const {
    DVector<double> result;
    result.resize(locs.rows());
    ADT<2,2,1> engine(model_.pde().domain());
    
    for(std::size_t i = 0; i < locs.rows(); ++i){
      // search element containing point
      SVector<2> p;
      p << locs.row(i)[0], locs.row(i)[1];
      auto e = engine.locate(p);
      // compute value of field at point
      double v = std::numeric_limits<double>::quiet_NaN();
      if(e != nullptr){
	v = 0;
	// evaluate the solution at point
	for(size_t j = 0; j < 3; ++j){
	  v += coeff(e->nodeIDs()[j],0) * model_.pde().basis()[e->ID()][j](p);
	}
      }
      result[i] = v;
    }
    return result;
  }
  
};

// definition of Rcpp module
typedef R_STRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> STRPDE_Laplacian_2D_GeoStatNodes;
RCPP_MODULE(STRPDE_Laplacian_2D_GeoStatNodes) {
  Rcpp::class_<STRPDE_Laplacian_2D_GeoStatNodes>("STRPDE_Laplacian_2D_GeoStatNodes")
    .constructor<Laplacian_2D_Order1>()
    // getters
    .method("fitted",         &STRPDE_Laplacian_2D_GeoStatNodes::fitted)
    // setters
    .method("set_lambda_s",     &STRPDE_Laplacian_2D_GeoStatNodes::set_lambda_s)
    .method("set_lambda_t",     &STRPDE_Laplacian_2D_GeoStatNodes::set_lambda_t)
    .method("set_time_domain",  &STRPDE_Laplacian_2D_GeoStatNodes::set_time_domain)
    .method("set_time_locations",  &STRPDE_Laplacian_2D_GeoStatNodes::set_time_locations)
    .method("set_observations", &STRPDE_Laplacian_2D_GeoStatNodes::set_observations)
    .method("update_missing", &STRPDE_Laplacian_2D_GeoStatNodes::update_missing)
    .method("R0",       &STRPDE_Laplacian_2D_GeoStatNodes::R0)
    .method("eval_to_locs",      &STRPDE_Laplacian_2D_GeoStatNodes::eval_to_locs)
    .method("mass",       &STRPDE_Laplacian_2D_GeoStatNodes::mass)
    .method("Psi", &STRPDE_Laplacian_2D_GeoStatNodes::Psi)
    .method("init", &STRPDE_Laplacian_2D_GeoStatNodes::init)
    // solve method
    .method("solve",            &STRPDE_Laplacian_2D_GeoStatNodes::solve);
}

typedef R_STRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> STRPDE_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(STRPDE_Laplacian_2D_GeoStatLocations) {
  Rcpp::class_<STRPDE_Laplacian_2D_GeoStatLocations>("STRPDE_Laplacian_2D_GeoStatLocations")
    .constructor<Laplacian_2D_Order1>()
    // getters
    .method("fitted",         &STRPDE_Laplacian_2D_GeoStatLocations::fitted)
    // setters
    .method("set_lambda_s",     &STRPDE_Laplacian_2D_GeoStatLocations::set_lambda_s)
    .method("set_lambda_t",     &STRPDE_Laplacian_2D_GeoStatLocations::set_lambda_t)
    .method("set_time_domain",  &STRPDE_Laplacian_2D_GeoStatLocations::set_time_domain)
    .method("set_time_locations",  &STRPDE_Laplacian_2D_GeoStatLocations::set_time_locations)
    .method("set_observations", &STRPDE_Laplacian_2D_GeoStatLocations::set_observations)
    .method("update_missing", &STRPDE_Laplacian_2D_GeoStatLocations::update_missing)
    .method("set_locations", &STRPDE_Laplacian_2D_GeoStatLocations::set_locations)
    .method("R0",       &STRPDE_Laplacian_2D_GeoStatLocations::R0)
    .method("eval_to_locs",      &STRPDE_Laplacian_2D_GeoStatLocations::eval_to_locs)
    .method("mass",       &STRPDE_Laplacian_2D_GeoStatLocations::mass)
    .method("Psi", &STRPDE_Laplacian_2D_GeoStatLocations::Psi)
    .method("init", &STRPDE_Laplacian_2D_GeoStatLocations::init)
    // solve method
    .method("solve",            &STRPDE_Laplacian_2D_GeoStatLocations::solve);
}
