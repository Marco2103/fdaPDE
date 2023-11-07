#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/IO/CSVReader.h"
#include "../fdaPDE/core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
#include "../fdaPDE/core/FEM/operators/SpaceVaryingFunctors.h"
using fdaPDE::core::FEM::SpaceVaryingDiffusion;
using fdaPDE::core::FEM::SpaceVaryingAdvection;
#include "core/MESH/Mesh.h"
#include "../fdaPDE/models/regression/STRPDE.h"
using fdaPDE::models::STRPDE;

#include "../fdaPDE/models/regression/SQRPDE.h"
using fdaPDE::models::SQRPDE;

#include "../fdaPDE/models/SamplingDesign.h"
#include "../fdaPDE/calibration/GCV.h"
using fdaPDE::calibration::GCV;
using fdaPDE::calibration::ExactGCV;
using fdaPDE::calibration::ExactEDF;
using fdaPDE::calibration::StochasticEDF;
#include "../fdaPDE/core/OPT/optimizers/GridOptimizer.h"
using fdaPDE::core::OPT::GridOptimizer;
#include "../fdaPDE/core/OPT/optimizers/Newton.h"
using fdaPDE::core::OPT::NewtonOptimizer;

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

#include "../../fdaPDE/preprocess/InitialConditionEstimator.h"
using fdaPDE::preprocess::InitialConditionEstimator;

// for time and memory performances
#include <chrono>
#include <iomanip>
using namespace std::chrono;
#include <unistd.h>
#include <fstream>




/* test 1 SQRPDE - Time
   domain:       unit square [0,1] x [0,1] (coarse)
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   GCV optimization: grid exact
 */
TEST(GCV_SQRPDE_Time, Test1_Laplacian_NonParametric_GeostatisticalAtNodes_GridExact) {
  
  
  // Parameters 
  const std::string TestNumber = "1"; 
  
  std::vector<double> alphas = {0.1, 0.5, 0.9}; 

  // number of simulations
  unsigned int n_sim = 6; 

  // Marco
  // std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  // Ilenia 
  std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared"; 

  std::string path_test = path + "/space_time/Test_" + TestNumber ;


  // define time domain
  DVector<double> time_mesh;
  time_mesh.resize(11);
  std::size_t i = 0;
  for(double x = 0; x <= 2; x+=0.2, ++i) time_mesh[i] = x;
  
  // define spatial domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square_coarse");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3*time_mesh.rows(), 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  for(double alpha : alphas){

    unsigned int alpha_int = alpha*100; 

    std::cout << "------------------------------------------alpha=" << std::to_string(alpha_int) << "%-------------------------------------------------" << std::endl; 

  
    // define statistical model
    SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatMeshNodes,
    fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);


    for(unsigned int sim = 1; sim <= n_sim; ++sim){

      // load data from .csv files
      CSVReader<double> reader{};
      CSVFile<double> yFile; // observation file
      yFile = reader.parseFile(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/y.csv");
      DMatrix<double> y = yFile.toEigen();

      // set model data
      BlockFrame<double, int> df;
      df.stack(OBSERVATIONS_BLK, y);   

      model.setData(df);


      model.init(); // init model per la PDE

      std::vector<SVector<2>> lambdas_t;
      // for(double x = -6.0; x <= -5.0; x +=0.20) lambdas_t.push_back(SVector<2>(std::pow(10,x), std::pow(10,x)));  // 10, 90

      for(double x = -7.0; x <= -3.6; x +=0.20) lambdas_t.push_back(SVector<2>(std::pow(10,x), std::pow(10,x)));  // 50%

      
      // define GCV function and optimize
      GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
      GridOptimizer<2> opt;

      ScalarField<2, decltype(GCV)> obj(GCV);
      opt.optimize(obj, lambdas_t); // optimize gcv field
      SVector<2> best_lambda = opt.optimum();
      
      std::cout << "Best lambda = " << best_lambda << std::endl ; 

      // Save Lambda opt
      std::ofstream fileLambdaoptS(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/LambdaS.csv");
      if (fileLambdaoptS.is_open()){
      fileLambdaoptS << std::setprecision(16) << best_lambda[0];
      fileLambdaoptS.close();
      }

      std::ofstream fileLambdaoptT(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/LambdaT.csv");
      if (fileLambdaoptT.is_open()){
      fileLambdaoptT << std::setprecision(16) << best_lambda[1];
      fileLambdaoptT.close();
      }
           
      
      // Save GCV scores
      std::ofstream fileGCV_scores(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/GCV_scores.csv");
      for(std::size_t i = 0; i < GCV.values().size(); ++i) 
        fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

      fileGCV_scores.close(); 

      // Save edf
      std::ofstream file_edf(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/GCV_edf.csv");
      for(std::size_t i = 0; i < GCV.values().size(); ++i) 
        file_edf << std::setprecision(16) << std::sqrt(GCV.edfs()[i]) << "\n" ; 

      file_edf.close(); 

    }

  }


}



/* test 1 SQRPDE - Time
   domain:       unit square [0,1] x [0,1] (coarse)
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   GCV optimization: grid stoch
 */
TEST(GCV_SQRPDE_Time, Test1_Laplacian_NonParametric_GeostatisticalAtNodes_GridStoch) {
  
  
  // Parameters 
  const std::string TestNumber = "1"; 

  double alpha = 0.90;
  unsigned int alpha_int = alpha*100; 
  const std::string alpha_string = std::to_string(alpha_int);
  
  // Marco
  // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  // Ilenia 
  std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared"; 

  std::string path_test = path + "/space_time/Test_" + TestNumber ;


  // define time domain
  DVector<double> time_mesh;
  time_mesh.resize(11);
  std::size_t i = 0;
  for(double x = 0; x <= 2; x+=0.2, ++i) time_mesh[i] = x;
  
  // define spatial domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square_coarse");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3*time_mesh.rows(), 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  
  // define statistical model
  SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatMeshNodes,
	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);


  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile(path_test + "/z.csv");
  DMatrix<double> y = yFile.toEigen();

  // set model data
  BlockFrame<double, int> df;
  df.stack(OBSERVATIONS_BLK, y);   

  model.setData(df);


  model.init(); // init model per la PDE

  std::vector<SVector<2>> lambdas_t;
  for(double x = -6.0; x <= -5.0; x +=0.25) lambdas_t.push_back(SVector<2>(std::pow(10,x), std::pow(10,x)));

  
  // define GCV function and optimize
  std::size_t seed = 438172;
  GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 100, seed);
  GridOptimizer<2> opt;

  ScalarField<2, decltype(GCV)> obj(GCV);
  opt.optimize(obj, lambdas_t); // optimize gcv field
  SVector<2> best_lambda = opt.optimum();

  
  std::cout << "Best lambda = " << best_lambda << std::endl ; 

  // Save Lambda opt
  std::ofstream fileLambdaopt(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/LambdaCpp.csv");
  for(std::size_t i = 0; i < best_lambda.size(); ++i) 
    fileLambdaopt << std::setprecision(16) << best_lambda[i] << "\n" ; 
    
  fileLambdaopt.close();
  
  
  // Save GCV scores
  std::ofstream fileGCV_scores(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/GCV_scores.csv");
  for(std::size_t i = 0; i < GCV.values().size(); ++i) 
    fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

  fileGCV_scores.close(); 

  // Save edf
  std::ofstream file_edf(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/GCV_edf.csv");
  for(std::size_t i = 0; i < GCV.values().size(); ++i) 
    file_edf << std::setprecision(16) << std::sqrt(GCV.edfs()[i]) << "\n" ; 

  file_edf.close(); 


}


/* test 2 
   domain:       c-shaped
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
TEST(GCV_SQRPDE_Time, Test2_Laplacian_SemiParametric_GeostatisticalAtLocations_Separable_Monolithic_GridExact) {

  // Parameters 
  const std::string TestNumber = "2"; 
  
  // Marco
  // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  // Ilenia 
  std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared"; 

  std::string path_test = path + "/space_time/Test_" + TestNumber ;

  std::vector<double> alphas = {0.1, 0.9};  // 0.5, 

  // number of simulations
  unsigned int n_sim = 10; 


  // define time domain
  DVector<double> time_mesh;
  time_mesh.resize(5);
  for(std::size_t i = 0; i < 5; ++i)
    time_mesh[i] = (fdaPDE::testing::pi/4)*i;

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("c_shaped");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // load sample position
  CSVReader<double> reader{};
  CSVFile<double> locFile; // locations file
  locFile = reader.parseFile(path_test + "/locs.csv");
  DMatrix<double> loc = locFile.toEigen();
  
  // load covariates from .csv files
  CSVFile<double> XFile; // design matrix
  XFile = reader.parseFile  (path_test + "/X.csv");
  DMatrix<double> X = XFile.toEigen();

  for(double alpha : alphas){

    unsigned int alpha_int = alpha*100; 

    std::cout << "------------------------------------------alpha=" << std::to_string(alpha_int) << "%-------------------------------------------------" << std::endl; 

    // Define model 
    SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatLocations,
      fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);

    for(unsigned int sim = 1; sim <= n_sim; ++sim){

      // load data from .csv files
      CSVFile<double> yFile; // design matrix
      yFile = reader.parseFile  (path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/y.csv");
      DMatrix<double> y = yFile.toEigen();

      // set model data
      BlockFrame<double, int> df;
      df.stack (OBSERVATIONS_BLK,  y);
      df.stack (DESIGN_MATRIX_BLK, X);
      model.set_spatial_locations(loc);
      model.setData(df);

      // solve smoothing problem
      model.init();

      // Define vector of lambdas
      std::vector<SVector<2>> lambdas_t;
      // for(double x = -3.8; x <= -2.6; x +=0.1) lambdas_t.push_back(SVector<2>(std::pow(10,x), std::pow(10,x)));   // 50%
      for(double x = -4.8; x <= -3.0; x +=0.1) lambdas_t.push_back(SVector<2>(std::pow(10,x), std::pow(10,x)));  // 10% and 90%

      // define GCV function and optimize
      GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
      GridOptimizer<2> opt;

      ScalarField<2, decltype(GCV)> obj(GCV);
      opt.optimize(obj, lambdas_t); // optimize gcv field
      SVector<2> best_lambda = opt.optimum();

      std::cout << "Best lambda = " << best_lambda << std::endl ; 

      // Save Lambda opt
      std::ofstream fileLambdaoptS(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/LambdaS.csv");
      if (fileLambdaoptS.is_open()){
      fileLambdaoptS << std::setprecision(16) << best_lambda[0];
      fileLambdaoptS.close();
      }

      std::ofstream fileLambdaoptT(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/LambdaT.csv");
      if (fileLambdaoptT.is_open()){
      fileLambdaoptT << std::setprecision(16) << best_lambda[1];
      fileLambdaoptT.close();
      }


      // Save GCV scores
      std::ofstream fileGCV_scores(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/GCV_scores.csv");
      for(std::size_t i = 0; i < GCV.values().size(); ++i) 
        fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

      fileGCV_scores.close(); 

      // Save edf
      std::ofstream file_edf(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/GCV_edf.csv");
      for(std::size_t i = 0; i < GCV.values().size(); ++i) 
        file_edf << std::setprecision(16) << std::sqrt(GCV.edfs()[i]) << "\n" ; 

      file_edf.close(); 

    }
  }
  
}




/* test 2 
   domain:       c-shaped
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
TEST(GCV_SQRPDE_Time, Test2_Laplacian_SemiParametric_GeostatisticalAtLocations_Separable_Monolithic_GridStoch) {

  // Parameters 
  const std::string TestNumber = "2"; 

  double alpha = 0.50;
  unsigned int alpha_int = alpha*100; 
  const std::string alpha_string = std::to_string(alpha_int);
  
  // Marco
  // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  // Ilenia 
  std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared"; 

  std::string path_test = path + "/space_time/Test_" + TestNumber ;


  // define time domain
  DVector<double> time_mesh;
  time_mesh.resize(5);
  for(std::size_t i = 0; i < 5; ++i)
    time_mesh[i] = (fdaPDE::testing::pi/4)*i;

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("c_shaped");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // Define model 
  SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatLocations,
	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);


  // load sample position
  CSVReader<double> reader{};
  CSVFile<double> locFile; // locations file
  locFile = reader.parseFile(path_test + "/locs.csv");
  DMatrix<double> loc = locFile.toEigen();
  model.set_spatial_locations(loc);
  
  // load data from .csv files
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile  (path_test + "/z.csv");
  DMatrix<double> y = yFile.toEigen();
  CSVFile<double> XFile; // design matrix
  XFile = reader.parseFile  (path_test + "/X.csv");
  DMatrix<double> X = XFile.toEigen();

  // set model data
  BlockFrame<double, int> df;
  df.stack (OBSERVATIONS_BLK,  y);
  df.stack (DESIGN_MATRIX_BLK, X);
  model.setData(df);
  
  // solve smoothing problem
  model.init();

  // Define vector of lambdas
  std::vector<SVector<2>> lambdas_t;
  for(double x = -4.25; x <= -3.25; x +=0.25) lambdas_t.push_back(SVector<2>(std::pow(10,x), std::pow(10,x)));

  // define GCV function and optimize
  std::size_t seed = 438172;
  GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed);   // change r
  GridOptimizer<2> opt;

  ScalarField<2, decltype(GCV)> obj(GCV);
  opt.optimize(obj, lambdas_t); // optimize gcv field
  SVector<2> best_lambda = opt.optimum();
  
  std::cout << "Best lambda = " << best_lambda << std::endl ; 

  // Save Lambda opt
  // std::ofstream fileLambdaopt(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/LambdaCpp_Wood.csv");
  std::ofstream fileLambdaopt(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/LambdaCpp.csv");
  for(std::size_t i = 0; i < best_lambda.size(); ++i) 
    fileLambdaopt << std::setprecision(16) << best_lambda[i] << "\n" ; 
    
  fileLambdaopt.close();
  
  
  // Save GCV scores
  // std::ofstream fileGCV_scores(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/GCV_scores_Wood.csv");
  std::ofstream fileGCV_scores(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/GCV_scores.csv");
  for(std::size_t i = 0; i < GCV.values().size(); ++i) 
    fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

  fileGCV_scores.close(); 

  // Save edf
  // std::ofstream file_edf(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/GCV_edf_Wood.csv");
  std::ofstream file_edf(path_test + "/alpha_" + alpha_string + "/GCV/Stoch/GCV_edf.csv");
  for(std::size_t i = 0; i < GCV.values().size(); ++i) 
    file_edf << std::setprecision(16) << std::sqrt(GCV.edfs()[i]) << "\n" ; 

  file_edf.close(); 

}
  
