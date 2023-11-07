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
#include "../fdaPDE/models/regression/MSQRPDE.h"
using fdaPDE::models::MSQRPDE;
#include "../fdaPDE/models/SamplingDesign.h"
#include "../../fdaPDE/models/regression/Distributions.h"

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

#include<fstream>
#include<iostream>


// /* test 1
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//  */
// TEST(MSQRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes) {

//   // path test  
//   std::string C_path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/2D_test1"; 
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/multiple_quantiles/Tests/Test_1"; 

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_44");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u);  // definition of regularizing PDE

//   // define statistical model
//   std::vector<double> alphas = {0.01, 0.05, 0.10, 0.90, 0.95, 0.99}; 

//   const std::string data_type = "hetero"; 

//   // Simulations 
//   const unsigned int M = 10; 

//   for(auto m = 1; m <= M; ++m){

//     MSQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alphas);
//     // use optimal lambda to avoid possible numerical issues
//     CSVReader<double> reader{};
//     CSVFile<double> lFile; // lambdas file
//     lFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/lambdas_opt.csv");
//     DMatrix<double> lambdas = lFile.toEigen();   // the vector should be saved with the "R format"

//     model.setLambdas_S(lambdas);
//     // load data from .csv files
//     CSVFile<double> yFile; // observation file
//     yFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/z.csv");
//     DMatrix<double> y = yFile.toEigen();

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     model.setData(df);

//     // solve smoothing problem
//     model.init();
//     model.solve();

//     // Save solution
//     DMatrix<double> computedF = model.f();
//     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filef(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/f_all.csv");
//     if (filef.is_open()){
//       filef << computedF.format(CSVFormatf);
//       filef.close();
//     }

//     // // debug 
//     // DMatrix<double> computedA = model.A_mult();
//     // const static Eigen::IOFormat CSVFormatA(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileA(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/A.csv");
//     // if (fileA.is_open()){
//     //   fileA << computedA.format(CSVFormatA);
//     //   fileA.close();
//     // }

//     // DMatrix<double> computedW = model.W_mult();
//     // const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileW(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/W.csv");
//     // if (fileW.is_open()){
//     //   fileW << computedW.format(CSVFormatW);
//     //   fileW.close();
//     // }

//     // DMatrix<double> computedWbar = model.Wbar_mult();
//     // const static Eigen::IOFormat CSVFormatWbar(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileWbar(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/Wbar.csv");
//     // if (fileWbar.is_open()){
//     //   fileWbar << computedWbar.format(CSVFormatWbar);
//     //   fileWbar.close();
//     // }

//     // DMatrix<double> computedDelta = model.Delta_mult();
//     // const static Eigen::IOFormat CSVFormatDelta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileDelta(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/Delta.csv");
//     // if (fileDelta.is_open()){
//     //   fileDelta << computedDelta.format(CSVFormatDelta);
//     //   fileDelta.close();
//     // }

//     // DMatrix<double> computedD_script = model.D_script();
//     // const static Eigen::IOFormat CSVFormatD_script(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileD_script(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/D_script.csv");
//     // if (fileD_script.is_open()){
//     //   fileD_script << computedD_script.format(CSVFormatD_script);
//     //   fileD_script.close();
//     // }

//     // DMatrix<double> computedDscriptj = model.Dscriptj();
//     // const static Eigen::IOFormat CSVFormatDscriptj(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileDscriptj(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/D_script_j.csv");
//     // if (fileDscriptj.is_open()){
//     //   fileDscriptj << computedDscriptj.format(CSVFormatDscriptj);
//     //   fileDscriptj.close();
//     // }

//     // DMatrix<double> computedPsi = model.Psi_mult();
//     // const static Eigen::IOFormat CSVFormatPsi(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream filePsi(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/Psi.csv");
//     // if (filePsi.is_open()){
//     //   filePsi << computedPsi.format(CSVFormatPsi);
//     //   filePsi.close();
//     // }

//   }

// }


// /* test 2
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//  */
// TEST(MSQRPDE, Test2_Laplacian_NonParametric_GeostatisticalAtNodes) {

//   // path test  
//   std::string C_path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/2D_test1"; 
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/multiple_quantiles/Tests/Test_2"; 

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_32");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u);  // definition of regularizing PDE

//   // define statistical model
//   std::vector<double> alphas = {0.01, 0.05, 0.10, 0.90, 0.95, 0.99}; 

//   const std::string data_type = "hetero"; 

//   // // Read covariates
//   CSVReader<double> reader{};
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile (R_path + "/data_" + data_type + "/X.csv");
//   DMatrix<double> X = XFile.toEigen();


//   // Simulations 
//   const unsigned int M = 10; 

//   for(auto m = 1; m <= M; ++m){

//     std::cout << "--------------------Simulation #" << std::to_string(m) << "-------------" << std::endl; 

//     MSQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alphas);
//     // use optimal lambda to avoid possible numerical issues
//     CSVFile<double> lFile; // lambdas file
//     lFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/lambdas_opt.csv");
//     DMatrix<double> lambdas = lFile.toEigen();   // the vector should be saved with the "R format"

//     model.setLambdas_S(lambdas);
//     // load data from .csv files
//     CSVFile<double> yFile; // observation file
//     yFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/z.csv");
//     DMatrix<double> y = yFile.toEigen();

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     df.insert(DESIGN_MATRIX_BLK, X);
//     model.setData(df);

//     // solve smoothing problem
//     model.init();
//     model.solve();

//     // Save solution
//     DMatrix<double> computedF = model.f();
//     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filef(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/f_all.csv");
//     if (filef.is_open()){
//       filef << computedF.format(CSVFormatf);
//       filef.close();
//     }

//     DMatrix<double> computedBeta = model.beta();
//     const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream fileBeta(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/beta_all.csv");
//     if (fileBeta.is_open()){
//       fileBeta << computedBeta.format(CSVFormatBeta);
//       fileBeta.close();
//     }

//     // // Debug
//     // DMatrix<double> computedDelta = model.Delta_mult();
//     // const static Eigen::IOFormat CSVFormatDelta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileDelta(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/Delta.csv");
//     // if (fileDelta.is_open()){
//     //   fileDelta << computedDelta.format(CSVFormatDelta);
//     //   fileDelta.close();
//     // }

//     // DMatrix<double> computedW = model.W_mult();
//     // std::cout << "size:  " << computedW.rows() << " " << computedW.cols() << std::endl; 
//     // const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileW(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/W.csv");
//     // if (fileW.is_open()){
//     //   fileW << computedW.format(CSVFormatW);
//     //   fileW.close();
//     // }

//     // DMatrix<double> computedXtWX = model.XtWX_multiple();
//     // const static Eigen::IOFormat CSVFormatXtWX(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileXtWX(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/XtWX.csv");
//     // if (fileXtWX.is_open()){
//     //   fileXtWX << computedXtWX.format(CSVFormatXtWX);
//     //   fileXtWX.close();
//     // }

//     // DMatrix<double> computedH = model.H_mult_debug();
//     // const static Eigen::IOFormat CSVFormatH(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream fileH(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/H.csv");
//     // if (fileH.is_open()){
//     //   fileH << computedH.format(CSVFormatH);
//     //   fileH.close();
//     // }
//   }
// }


// /* test 3 
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//  */
// TEST(MSQRPDE, Test3_Adv_SemiParametric_GeostatisticalAtLocations_GridExact) {

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

//     MSQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alphas);
//     model.set_spatial_locations(loc);

//     // use optimal lambda to avoid possible numerical issues
//     CSVFile<double> lFile; // lambdas file
//     lFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/single_est/lambdas_opt.csv");
//     DMatrix<double> lambdas = lFile.toEigen();   // the vector should be saved with the "R format"

//     model.setLambdas_S(lambdas);
//     // load data from .csv files
//     CSVFile<double> yFile; // observation file
//     yFile = reader.parseFile(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/z.csv");
//     DMatrix<double> y = yFile.toEigen();

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     df.insert(DESIGN_MATRIX_BLK, X);
//     model.setData(df);

//     // solve smoothing problem
//     model.init();
//     model.solve();

//     // Save solution
//     DMatrix<double> computedF = model.f();
//     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filef(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/f_all.csv");
//     if (filef.is_open()){
//       filef << computedF.format(CSVFormatf);
//       filef.close();
//     }

//     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filefn(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/fn_all.csv");
//     if (filefn.is_open()){
//       filefn << computedFn.format(CSVFormatfn);
//       filefn.close();
//     }

//     DMatrix<double> computedBeta = model.beta();
//     const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream fileBeta(R_path + "/data_" + data_type + "/sim_" + std::to_string(m) + "/mult_est/beta_all.csv");
//     if (fileBeta.is_open()){
//       fileBeta << computedBeta.format(CSVFormatBeta);
//       fileBeta.close();
//     }

//   }
// }




