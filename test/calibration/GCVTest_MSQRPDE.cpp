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
#include "../fdaPDE/models/regression/SRPDE.h"
using fdaPDE::models::SRPDE;
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


// for time and memory performances
#include <chrono>
#include <iomanip>
using namespace std::chrono;
#include <unistd.h>
#include <fstream>


// /* test 1     
//    domain:       unit square [0,1] x [0,1] (coarse)
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test9_Laplacian_NonParametric_GeostatisticalAtNodes_GridExact) {

//   // path test  
//   std::string C_path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/2D_test1"; 
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/multiple_quantiles/Tests/Test_1"; 

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_44"); 
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   std::vector<double> alphas = {0.01, 0.05, 0.10, 0.90, 0.95, 0.99}; 

//   const std::string data_type = "hetero"; 

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -7.2; x <= -6.3; x +=0.1) lambdas.push_back(SVector<1>(std::pow(10,x)));
//   DVector<double> best_lambda;
//   best_lambda.resize(alphas.size());  

//   // Simulations 
//   const unsigned int M = 10; 

//   for(auto m = 1; m <= M; ++m){
//     unsigned int ind = 0; 
//     std::ofstream fileGCV(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/gcv_scores.csv");
//     for(auto alpha : alphas){

//         std::cout << "------------------alpha=" << std::to_string(alpha) << "-----------------" << std::endl; 

//         SQRPDE<decltype(problem), fdaPDE::models::SpaceOnly, 
//               fdaPDE::models::GeoStatMeshNodes, fdaPDE::models::MonolithicSolver> model(problem, alpha);

//         // load data from .csv files
//         CSVReader<double> reader{};
//         CSVFile<double> yFile; // observation file
//         yFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/z.csv");
//         DMatrix<double> y = yFile.toEigen();

//         // set model data
//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         model.setData(df);

//         model.init(); // init model

//         // define GCV function and optimize
//         GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//         GridOptimizer<1> opt;

//         ScalarField<1, decltype(GCV)> obj(GCV);
//         opt.optimize(obj, lambdas); // optimize gcv field
//         std::cout << "opt: " << opt.optimum()[0] << std::endl; 
//         best_lambda[ind] = opt.optimum()[0];
         
//         std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda[ind] << std::endl; 
//         ind++;

//         // gcv scores
//         for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//           fileGCV << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n"; 

//       }

//       const static Eigen::IOFormat CSVFormatL(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//       std::ofstream fileL(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/lambdas_opt.csv");
//       if (fileL.is_open()){
//         fileL << best_lambda.format(CSVFormatL);
//         fileL.close();
//       }

//       fileGCV.close(); 
//   }

// }

// /* test 2 
//    domain:       unit square [0,1] x [0,1] (coarse)
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test9bis_Laplacian_NonParametric_GeostatisticalAtNodes_GridExact) {

//   // path test  
//   std::string C_path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/2D_test2"; 
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/multiple_quantiles/Tests/Test_2"; 

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_32"); 
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   std::vector<double> alphas = {0.01, 0.05, 0.10, 0.90, 0.95, 0.99}; 

//   const std::string data_type = "hetero"; 

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.9; x <= -5.8; x +=0.1) lambdas.push_back(SVector<1>(std::pow(10,x)));
//   DVector<double> best_lambda;
//   best_lambda.resize(alphas.size());  

//   // Read covariates
//   CSVReader<double> reader{};
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile (R_path + "/data_" + data_type + "/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // Simulations 
//   const unsigned int M = 10; 

//   for(auto m = 1; m <= M; ++m){
//     std::cout << "--------------------Simulation #" << std::to_string(m) << "-------------" << std::endl; 
//     unsigned int ind = 0; 
//     std::ofstream fileGCV(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/gcv_scores.csv");
//     for(auto alpha : alphas){

//         std::cout << "------------------alpha=" << std::to_string(alpha) << "-----------------" << std::endl; 

//         SQRPDE<decltype(problem), fdaPDE::models::SpaceOnly, 
//               fdaPDE::models::GeoStatMeshNodes, fdaPDE::models::MonolithicSolver> model(problem, alpha);

//         // load data from .csv files
//         CSVFile<double> yFile; // observation file
//         yFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/z.csv");
//         DMatrix<double> y = yFile.toEigen();

//         // set model data
//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(DESIGN_MATRIX_BLK, X);
//         model.setData(df);

//         model.init(); // init model

//         // define GCV function and optimize
//         GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//         GridOptimizer<1> opt;

//         ScalarField<1, decltype(GCV)> obj(GCV);
//         opt.optimize(obj, lambdas); // optimize gcv field
//         std::cout << "opt: " << opt.optimum()[0] << std::endl; 
//         best_lambda[ind] = opt.optimum()[0];
         
//         std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda[ind] << std::endl; 
//         ind++;

//         // gcv scores
//         for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//           fileGCV << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n"; 

//       }

//       const static Eigen::IOFormat CSVFormatL(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//       std::ofstream fileL(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/lambdas_opt.csv");
//       if (fileL.is_open()){
//         fileL << best_lambda.format(CSVFormatL);
//         fileL.close();
//       }

//       fileGCV.close(); 
//   }

// }




// /* Test 3 (individual estimations)
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: space varying PDE 
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(single_MSQRPDE, Test_Adv_SemiParametric_GeostatisticalAtLocations_GridExact) {

//   // path test  
//   std::string C_path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/2D_test3"; 
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/multiple_quantiles/Tests/Test_3"; 

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped_631"); 

//   // load PDE coefficients data
//   CSVReader<double> reader{};
//   CSVFile<double> adveFile; 
//   adveFile = reader.parseFile(C_path + "/b.csv");
//   DMatrix<double> adveData = adveFile.toEigen();
//   SpaceVaryingAdvection<2> adveCoeff;
//   adveCoeff.setData(adveData);
//   auto L = Laplacian() + Gradient(adveCoeff.asParameter());

//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   std::vector<double> alphas = {0.01, 0.05, 0.10, 0.90, 0.95, 0.99}; 

//   const std::string data_type = "hetero"; 

//   // Read covariates and locations
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile (R_path + "/data_" + data_type + "/X.csv");
//   DMatrix<double> X = XFile.toEigen();
//   CSVFile<double> locFile;
//   locFile = reader.parseFile(R_path + "/data_" + data_type + "/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   // Simulations 
//   const unsigned int M = 10; 

//   for(auto m = 1; m <= M; ++m){
//     std::cout << "--------------------Simulation #" << std::to_string(m) << "-------------" << std::endl; 
//     unsigned int alpha_ind = 0; 
//     std::ofstream fileGCV(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/gcv_scores.csv");
//     for(auto alpha : alphas){

//         unsigned int alpha_int = alpha*100; 
//         std::string alpha_string = std::to_string(alpha_int); 
//         std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

//         SQRPDE<decltype(problem), fdaPDE::models::SpaceOnly, 
//               fdaPDE::models::GeoStatLocations, fdaPDE::models::MonolithicSolver> model(problem, alpha);
//         model.set_spatial_locations(loc);

//         // use optimal lambda to avoid possible numerical issues
//         CSVFile<double> lFile; // lambdas file
//         lFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/lambdas_opt.csv");
//         double lambda = lFile.toEigen()(alpha_ind, 0);   // the vector should be saved with the "R format"
//         model.setLambdaS(lambda);

//         // load data from .csv files
//         CSVFile<double> yFile; // observation file
//         yFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/z.csv");
//         DMatrix<double> y = yFile.toEigen();

//         // set model data
//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(DESIGN_MATRIX_BLK, X);
//         model.setData(df);

//         model.init(); // init model
//         model.solve();

//         // Save solution
//         DMatrix<double> computedF = model.f();
//         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filef(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/f_" + alpha_string + ".csv");
//         if(filef.is_open()){
//           filef << computedF.format(CSVFormatf);
//           filef.close();
//         }
//         DMatrix<double> computedFn = model.Psi()*model.f();
//         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filefn(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/fn_" + alpha_string + ".csv");
//         if(filefn.is_open()){
//           filefn << computedFn.format(CSVFormatfn);
//           filefn.close();
//         }
//         DMatrix<double> computedBeta = model.beta();
//         const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream fileBeta(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/beta_" + alpha_string + ".csv");
//         if(fileBeta.is_open()){
//           fileBeta << computedBeta.format(CSVFormatBeta);
//           fileBeta.close();
//         }

//         alpha_ind++; 

//       }

//   }

// }


// /* Test 3 
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: space varying PDE 
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test_Adv_SemiParametric_GeostatisticalAtLocations_GridExact) {

//   // path test  
//   std::string C_path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/2D_test3"; 
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/multiple_quantiles/Tests/Test_3"; 

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped_631"); 

//   // load PDE coefficients data
//   CSVReader<double> reader{};
//   CSVFile<double> adveFile; 
//   adveFile = reader.parseFile(C_path + "/b.csv");
//   DMatrix<double> adveData = adveFile.toEigen();
//   SpaceVaryingAdvection<2> adveCoeff;
//   adveCoeff.setData(adveData);
//   auto L = Laplacian() + Gradient(adveCoeff.asParameter());

//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   std::vector<double> alphas = {0.01, 0.05, 0.10, 0.90, 0.95, 0.99}; 

//   const std::string data_type = "hetero"; 

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -5.0; x <= -2.8; x +=0.1) lambdas.push_back(SVector<1>(std::pow(10,x)));
//   DVector<double> best_lambda;
//   best_lambda.resize(alphas.size());  

//   // Read covariates and locations
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile (R_path + "/data_" + data_type + "/X.csv");
//   DMatrix<double> X = XFile.toEigen();
//   CSVFile<double> locFile;
//   locFile = reader.parseFile(R_path + "/data_" + data_type + "/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   // Simulations 
//   const unsigned int M = 10; 

//   for(auto m = 1; m <= M; ++m){
//     std::cout << "--------------------Simulation #" << std::to_string(m) << "-------------" << std::endl; 
//     unsigned int ind = 0; 
//     std::ofstream fileGCV(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/gcv_scores.csv");
//     for(auto alpha : alphas){

//         std::cout << "------------------alpha=" << std::to_string(alpha) << "-----------------" << std::endl; 

//         SQRPDE<decltype(problem), fdaPDE::models::SpaceOnly, 
//               fdaPDE::models::GeoStatLocations, fdaPDE::models::MonolithicSolver> model(problem, alpha);
//         model.set_spatial_locations(loc);

//         // load data from .csv files
//         CSVFile<double> yFile; // observation file
//         yFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/z.csv");
//         DMatrix<double> y = yFile.toEigen();

//         // set model data
//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         df.insert(DESIGN_MATRIX_BLK, X);
//         model.setData(df);

//         model.init(); // init model

//         // define GCV function and optimize
//         GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//         GridOptimizer<1> opt;

//         ScalarField<1, decltype(GCV)> obj(GCV);
//         opt.optimize(obj, lambdas); // optimize gcv field
//         std::cout << "opt: " << opt.optimum()[0] << std::endl; 
//         best_lambda[ind] = opt.optimum()[0];
         
//         std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda[ind] << std::endl; 
//         ind++;

//         // gcv scores
//         for(std::size_t i = 0; i < GCV.values().size(); ++i){
//           fileGCV << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n"; 
//         }

//       }

//       const static Eigen::IOFormat CSVFormatL(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//       std::ofstream fileL(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/lambdas_opt.csv");
//       if (fileL.is_open()){
//         fileL << best_lambda.format(CSVFormatL);
//         fileL.close();
//       }

//       fileGCV.close(); 
//   }

// }