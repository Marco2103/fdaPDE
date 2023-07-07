// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <thread>
#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/functional/fPCA.h>
using fdaPDE::models::FPCA;
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

// wrapper for SRPDE module
template <typename RegularizingPDE, typename S> class R_FPCA {
protected:
  typedef RegularizingPDE RegularizingPDE_;
  RegularizingPDE_ regularization_; // tipo di PDE

  // qui metti il modello che vuoi wrappare (FPLS??)
  FPCA<typename RegularizingPDE_::PDEType, fdaPDE::models::SpaceTimeSeparable, S, fdaPDE::models::gcv_lambda_selection> model_;
  BlockFrame<double, int> df_;

  std::shared_ptr<ThreadPool> tp;
public:
  
  R_FPCA(const RegularizingPDE_ &regularization)
    : regularization_(regularization), tp(std::make_shared<ThreadPool>(4)) {
    model_.setPDE(regularization_.pde());
    model_.init_multithreading(tp);
  };

  void init_pde() { model_.init_pde(); }
  
  /* setters */
  void set_lambda_s(double lambdaS) { model_.setLambdaS(lambdaS); }
  void set_lambda_t(double lambdaT) { model_.setLambdaT(lambdaT); }
  void set_time_domain(const DVector<double>& t) { model_.setTimeDomain(t); }
  void set_time_locations(const DMatrix<double>& time_locs) {
    model_.set_temporal_locations(time_locs);
  }
  void set_lambdas(std::vector<double> lambda_s, std::vector<double> lambda_t) {
    std::vector<SVector<2>> l_;
    for(std::size_t i = 0; i < lambda_s.size(); ++i)
      for(std::size_t j = 0; j < lambda_t.size(); ++j)
	l_.push_back(SVector<2>(lambda_s[i], lambda_t[j]));
    model_.setLambda(l_);
  }

  // void set_lambdas(std::vector<double> lambdas) {
  //   std::vector<SVector<1>> l_;
  //   for(auto v : lambdas) l_.push_back(SVector<1>(v));
  //   model_.setLambda(l_);
  // }

  void set_npc(std::size_t n) { model_.set_npc(n); }
  
  void set_observations(const DMatrix<double>& data) {
    df_.template insert<double>(OBSERVATIONS_BLK, data);
  }
  void set_locations(const DMatrix<double>& data) {
    model_.set_spatial_locations(data);
  }
  /* getters */
  DMatrix<double> loadings() const { return model_.loadings(); }
  DMatrix<double> scores() const { return model_.scores(); }

  SpMatrix<double> R0() const { return model_.R0(); }
  void init() { model_.init(); }
  void init_regularization() { model_.init_pde(); model_.init_regularization(); model_.init_sampling(); }
  SpMatrix<double> Psi() {
    return model_.Psi(not_nan());
  }

  void init_2() {
    model_.setData(df_);
    model_.init();
  }

  // find optimal smoothing vectors  (SOLO PER GCV)
  // void batch_solve(std::size_t n_batch) {
  //   model_.setData(df_);
  //   model_.batch_solve(n_batch);
  //   return;
  // }

  // void batch_solve_kcv(std::size_t n_batch) {
  //   model_.setData(df_);
  //   model_.batch_solve_kcv(n_batch);
  //   return;
  // }
  
  // void set_lambda(std::vector<double> lambdaS, std::vector<double> lambdaT) {
  //   std::vector<SVector<2>> lambdas;
  //   for(double s : lambdaS) {
  //     for(double t : lambdaT) {
  // 	lambdas.push_back( SVector<2>(s,t) );
  //     }
  //   }
  //   model_.setLambda(lambdas);
  //   return;
  // }
  
  /* initialize model and solve smoothing problem */
  double solve() {
    model_.setData(df_);
    model_.init();
    model_.solve();

    return 0; //model_.get_gcv();
  }

  Rcpp::List plot_basis(std::size_t i) {
    regularization_.pde().init();
    //regularization_.pde().solve();
    
    //Eigen::GeneralizedSelfAdjointEigenSolver<DMatrix<double>> ges;
    // compute generalized eigendecomposition of matrices R_1 and R_0:
    // find k \in \mathbb{R} and vector f such that R_1*f = k*R_0*f
    //ges.setMaxIterations(16);
    //ges.compute(DMatrix<double>(regularization_.pde().R1()), DMatrix<double>(regularization_.pde().R0()));

    EigenValueProblem<decltype(regularization_.pde())> evp(regularization_.pde());
    evp.solve();
    
    //std::cout << ges.eigenvalues() << std::endl;
    
    // define eigen solver
    // Eigen::GeneralizedEigenSolver<DMatrix<double>> ges;
    // // compute generalized eigendecomposition of matrices R_1 and R_0:
    // // find k \in \mathbb{R} and vector f such that R_1*f = k*R_0*f
    // ges.setMaxIterations(16);
    // ges.compute(DMatrix<double>(regularization_.pde().R1()), DMatrix<double>(regularization_.pde().R0()));

    // std::size_t idx = 0;
    // double min_eigen = ges.eigenvalues()[0];
    // for(std::size_t i = 1; i < ges.eigenvalues().rows(); ++i){
    //   if(ges.eigenvalues()[i] < min_eigen){
    // 	idx = i;
    // 	min_eigen = ges.eigenvalues()[idx];
    //   }	
    // }
    
    // DVector<double> result = ges.eigenvectors().col(i);
    DVector<double> result = evp.eigenfunctions().col(i);
    // result.resize(ges.eigenvectors().col(idx).rows());
    // for(std::size_t j = 0; j < ges.eigenvectors().col(idx).rows(); ++j)
    //   result[j] = ges.eigenvectors().col(idx).coeff(j,0);

    result = result;
    
    // LagrangianBasis<2,2,1> basis{};
    // double resolution = 0.01;
    // std::size_t NN = 1/resolution;
    // // print stuffs required by R
    // std::vector<double> x_coord{};
    // x_coord.resize(NN*NN);
    // std::vector<double> y_coord{};
    // y_coord.resize(NN*NN);
    // std::vector<double> data{};
    // data.resize(NN*NN);

    // // prepare for R layer
    // std::size_t j = 0;
    // std::size_t z = 0;
    // for(double x = 0; x < 1; x+=resolution, ++j){
    //   for(double y = 0; y < 1; y+=resolution, ++z){
    // 	x_coord[j*NN + z] = x;
    // 	y_coord[j*NN + z] = y;
    // 	if(1-x > y){
    // 	  data[j*NN + z] = basis[i](SVector<2>(x,y));
    // 	}else
    // 	  data[j*NN + z] = std::numeric_limits<double>::quiet_NaN();
    //   }
    //   z = 0;
    // }
    // return Rcpp::List::create(Rcpp::Named("x") = x_coord, Rcpp::Named("y") = y_coord, Rcpp::Named("solution") = data);

    return Rcpp::List::create(Rcpp::Named("eigenfunc") = evp.eigenfunctions(), Rcpp::Named("eigenval") = evp.eigenvalues());
  }

  DVector<double> explained_variance(const DMatrix<double>& scores) {
    Eigen::ColPivHouseholderQR<DMatrix<double>> s;
    s.compute(scores);

    return s.matrixR().diagonal();
  }
  
  Rcpp::List plot(double resolution, DMatrix<double> solution) const {
    // solution evaluation
    Evaluator<std::decay<decltype(regularization_.pde())>::type::local_dimension,
	      std::decay<decltype(regularization_.pde())>::type::embedding_dimension,
	      std::decay<decltype(regularization_.pde())>::type::basis_order> eval;
    fdaPDE::core::FEM::Raster<std::decay<decltype(regularization_.pde())>::type::local_dimension> img
      = eval.toRaster(model_.pde().domain(), solution, resolution, model_.pde().basis());
    // print stuffs required by R
    std::vector<double> x_coord{};
    x_coord.resize(img.size());
    std::vector<double> y_coord{};
    y_coord.resize(img.size());
    // prepare for R layer
    for(std::size_t i = 0; i < img.size(); ++i){
      x_coord[i] = img.coords_[i][0];
      y_coord[i] = img.coords_[i][1];
    }

    return Rcpp::List::create(Rcpp::Named("x") = x_coord,
			      Rcpp::Named("y") = y_coord,
			      Rcpp::Named("solution") = img.data_);
  }

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
typedef R_FPCA<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_Laplacian_2D_GeoStatNodes;
RCPP_MODULE(FPCA_Laplacian_2D_GeoStatNodes) {
  Rcpp::class_<FPCA_Laplacian_2D_GeoStatNodes>("FPCA_Laplacian_2D_GeoStatNodes")
    .constructor<Laplacian_2D_Order1>()
    // getters
    .method("loadings",         &FPCA_Laplacian_2D_GeoStatNodes::loadings)
    .method("scores",           &FPCA_Laplacian_2D_GeoStatNodes::scores)
    // setters
    .method("set_lambda_s",     &FPCA_Laplacian_2D_GeoStatNodes::set_lambda_s)
    .method("set_lambdas",     &FPCA_Laplacian_2D_GeoStatNodes::set_lambdas)
    .method("set_lambda_t",     &FPCA_Laplacian_2D_GeoStatNodes::set_lambda_t)
    .method("set_time_domain",  &FPCA_Laplacian_2D_GeoStatNodes::set_time_domain)
    .method("set_time_locations",  &FPCA_Laplacian_2D_GeoStatNodes::set_time_locations)
    .method("set_npc", &FPCA_Laplacian_2D_GeoStatNodes::set_npc)
    .method("set_observations", &FPCA_Laplacian_2D_GeoStatNodes::set_observations)
    .method("plot",             &FPCA_Laplacian_2D_GeoStatNodes::plot)
    .method("plot_basis",       &FPCA_Laplacian_2D_GeoStatNodes::plot_basis)
    .method("R0",       &FPCA_Laplacian_2D_GeoStatNodes::R0)
    .method("init",       &FPCA_Laplacian_2D_GeoStatNodes::init)
    .method("init_regularization",       &FPCA_Laplacian_2D_GeoStatNodes::init_regularization)
    .method("init_2",       &FPCA_Laplacian_2D_GeoStatNodes::init_2)
    .method("Psi",       &FPCA_Laplacian_2D_GeoStatNodes::Psi)
    // solve method
    .method("eval_to_locs",      &FPCA_Laplacian_2D_GeoStatNodes::eval_to_locs)
    // .method("batch_solve",      &FPCA_Laplacian_2D_GeoStatNodes::batch_solve)
    // .method("batch_solve_kcv",      &FPCA_Laplacian_2D_GeoStatNodes::batch_solve_kcv)
    .method("solve",            &FPCA_Laplacian_2D_GeoStatNodes::solve);
}

typedef R_FPCA<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> FPCA_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(FPCA_Laplacian_2D_GeoStatLocations) {
  Rcpp::class_<FPCA_Laplacian_2D_GeoStatLocations>("FPCA_Laplacian_2D_GeoStatLocations")
    .constructor<Laplacian_2D_Order1>()
    // getters
    .method("loadings",         &FPCA_Laplacian_2D_GeoStatLocations::loadings)
    .method("scores",           &FPCA_Laplacian_2D_GeoStatLocations::scores)
    // setters
    .method("set_lambda_s",     &FPCA_Laplacian_2D_GeoStatLocations::set_lambda_s)
    .method("set_lambdas",     &FPCA_Laplacian_2D_GeoStatLocations::set_lambdas)
    .method("set_lambda_t",     &FPCA_Laplacian_2D_GeoStatLocations::set_lambda_t)
    .method("set_time_domain",  &FPCA_Laplacian_2D_GeoStatLocations::set_time_domain)
    .method("set_time_locations",  &FPCA_Laplacian_2D_GeoStatLocations::set_time_locations)
    .method("set_npc", &FPCA_Laplacian_2D_GeoStatLocations::set_npc)
    .method("set_locations",  &FPCA_Laplacian_2D_GeoStatLocations::set_locations)
    .method("set_observations", &FPCA_Laplacian_2D_GeoStatLocations::set_observations)
    .method("plot",             &FPCA_Laplacian_2D_GeoStatLocations::plot)
    .method("plot_basis",       &FPCA_Laplacian_2D_GeoStatLocations::plot_basis)
    .method("R0",       &FPCA_Laplacian_2D_GeoStatLocations::R0)
    .method("init_regularization",       &FPCA_Laplacian_2D_GeoStatLocations::init_regularization)
    .method("init",       &FPCA_Laplacian_2D_GeoStatLocations::init)
    .method("init_2",       &FPCA_Laplacian_2D_GeoStatLocations::init_2)
    .method("Psi",       &FPCA_Laplacian_2D_GeoStatLocations::Psi)
    .method("explained_variance",       &FPCA_Laplacian_2D_GeoStatLocations::explained_variance)
    .method("eval_to_locs",      &FPCA_Laplacian_2D_GeoStatLocations::eval_to_locs)
    // solve method
    // .method("batch_solve",      &FPCA_Laplacian_2D_GeoStatLocations::batch_solve)
    // .method("batch_solve_kcv",      &FPCA_Laplacian_2D_GeoStatLocations::batch_solve_kcv)
    .method("solve",            &FPCA_Laplacian_2D_GeoStatLocations::solve);
}


typedef R_FPCA<Laplacian_3D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_Laplacian_3D_GeoStatNodes;
RCPP_MODULE(FPCA_Laplacian_3D_GeoStatNodes) {
  Rcpp::class_<FPCA_Laplacian_3D_GeoStatNodes>("FPCA_Laplacian_3D_GeoStatNodes")
    .constructor<Laplacian_3D_Order1>()
    // getters
    .method("loadings",         &FPCA_Laplacian_3D_GeoStatNodes::loadings)
    .method("scores",           &FPCA_Laplacian_3D_GeoStatNodes::scores)
    // setters
    .method("set_lambda_s",     &FPCA_Laplacian_3D_GeoStatNodes::set_lambda_s)
    .method("set_observations", &FPCA_Laplacian_3D_GeoStatNodes::set_observations)
    .method("plot",             &FPCA_Laplacian_3D_GeoStatNodes::plot)
    .method("plot_basis",       &FPCA_Laplacian_3D_GeoStatNodes::plot_basis)
    .method("init_regularization",       &FPCA_Laplacian_3D_GeoStatNodes::init_regularization)
    .method("init_pde",       &FPCA_Laplacian_3D_GeoStatNodes::init_pde)
    .method("R0",       &FPCA_Laplacian_3D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &FPCA_Laplacian_3D_GeoStatNodes::solve);
}
