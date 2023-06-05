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
#include "../fdaPDE/models/regression/SQRPDE.h"
using fdaPDE::models::SQRPDE;
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


/* test 1
   domain:       unit square
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   Data generation: as described in data/models/SRPDE/test1/README.md 

 */

// TEST(SQRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   double alpha = 0.5; 
//   const std::string alpha_string = "50"; 
//   const std::string TestNumber = "1"; 

//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   // yFile = reader.parseFile("data/models/SQRPDE/2D_test" + TestNumber + "/z.csv");
//   std::string data_macro_strategy_type = "skewed_data"; 
//   std::string data_strategy_type = "B"; 

//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/z.csv");             
//   DMatrix<double> y = yFile.toEigen();


//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   model.setData(df);

//   std::vector<double> seq_tol_weights = {0.00000001, 0.000001}; 
//   std::vector<std::string> seq_tol_weights_string = {"1e-08",  "1e-06"}; 

//   std::vector<double> seq_tol_FPIRLS = {0.000000001, 0.00000001, 0.0000001, 0.000001};
//   std::vector<std::string> seq_tol_FPIRLS_string = {"1e-09", "1e-08", "1e-07", "1e-06"}; 

//   std::string lin_sys_solver = "LU";    // depends on the "symmetry" option in R 

//   CSVFile<double> lambdaCSV; 
//   DMatrix<double> lambda; 

//   for(int i = 0; i < seq_tol_weights.size(); ++i ){
//     for(int j = 0; j < seq_tol_FPIRLS.size(); ++j){

//       // use optimal lambda to avoid possible numerical issues
//       lambdaCSV = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/" + lin_sys_solver + "/LambdaR_" + alpha_string + ".csv");     // from R 
      
//       lambda = lambdaCSV.toEigen();
//       model.setLambdaS(lambda(0,0));

//       // solve smoothing problem
//       model.setTolerances(seq_tol_weights[i], seq_tol_FPIRLS[j]); 
//       model.init();     
//       model.solve();

//       // Save C++ solution 
//       DMatrix<double> computedF = model.f();
//       const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//       // std::ofstream filef("data/models/SQRPDE/2D_test1/fnCpp_" + alpha_string + ".csv");
//       std::ofstream filef(R_path + "/R/Our/data/Test_" 
//                   + TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/" + lin_sys_solver + "/fnCpp_" + alpha_string + ".csv");

//       if (filef.is_open()){
//         filef << computedF.format(CSVFormatf);
//         filef.close();
//       }

//       // Matrix of weights at convergence 
//       DMatrix<double> W_conv = model.W(); 
//       const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//       std::ofstream file_W(R_path + "/R/Our/data/Test_" 
//                   + TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/" + lin_sys_solver + "/WCpp_" + alpha_string + ".csv");
//       if(file_W.is_open()){
//         file_W << W_conv.format(CSVFormatW) << '\n' ; 
//       }


//       double J = model.J_final_sqrpde();
//       std::ofstream fileJ(R_path + "/R/Our/data/Test_" 
//                   + TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/" + lin_sys_solver + "/JCpp.csv");
//       if (fileJ.is_open()){
//         fileJ << J;
//         fileJ.close();
//       }

//       std::size_t niter = model.niter_sqrpde();
//       std::ofstream filen(R_path + "/R/Our/data/Test_" 
//                   + TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/" + lin_sys_solver + "/niterCpp.csv");
//       if (filen.is_open()){
//         filen << niter;
//         filen.close();
//       }


//     }
//   }


// }

// /* test 2
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//  */


// TEST(SQRPDE, Test2_Laplacian_SemiParametric_GeostatisticalAtLocations) {
//   // define domain and regularizing PDE
  
//   double alpha = 0.1;  
//   double tol_weights = 1e-8;
//   double tol_FPIRLS = 1e-6;
//   std::string alpha_string = "10" ;
//   const std::string TestNumber = "2"; 
  
//   std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
//   std::vector<int> seq_n = {5502, 5254, 5038} ; 
//   std::vector<std::string> seq_n_string = {"5502", "5254", "5038"} ; 
  
//   CSVFile<double> lambdaCSV; 
//   DMatrix<double> lambda; 
//   CSVFile<double> yFile; 
//   CSVFile<double> XFile;
//   DMatrix<double> y;  
//   DMatrix<double> X;

//   for(int n = 0; n < seq_n.size(); ++n ){
//     std::string domain_type; 
//     if( n > 1000)
//       domain_type = "c_shaped_fine";
//     else
//       domain_type = "c_shaped_medium";

//     MeshLoader<Mesh2D<>> domain(domain_type);         // _" + seq_n_string[n]);
//     auto L = Laplacian();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//     PDE problem(domain.mesh, L, u); // definition of regularizing PDE
//     // define statistical model
//     SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);

//     CSVReader<double> reader{};
//     // load locations where data are sampled
//     CSVFile<double> locFile;
//     locFile = reader.parseFile(R_path + "/R/Our/data/Test_" 
//                     + TestNumber + "/alpha_" + alpha_string + "/n_" + seq_n_string[n] + "/locs.csv");
//     DMatrix<double> loc = locFile.toEigen();
  
//     lambdaCSV = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                     TestNumber + "/alpha_" + alpha_string + "/n_" + seq_n_string[n] + "/LambdaR_" + alpha_string + ".csv");     // from R 
        
//     lambda = lambdaCSV.toEigen();
//     model.setLambdaS(lambda(0,0));

//     model.set_spatial_locations(loc);

    
//     // load data from .csv files
//     yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                     TestNumber + "/alpha_" + alpha_string + "/n_" + seq_n_string[n] + "/z.csv"); 
//     y = yFile.toEigen();
//     XFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                     TestNumber + "/alpha_" + alpha_string + "/n_" + seq_n_string[n] + "/X.csv"); 
//     X = XFile.toEigen();

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK,  y);
//     df.insert(DESIGN_MATRIX_BLK, X);
    
//     model.setData(df);
//     model.setTolerances(tol_weights, tol_FPIRLS); 
//     // solve smoothing problem
//     model.init();
//     model.solve();



//     // Save C++ solution 
//     DMatrix<double> computedF = model.Psi()*model.f();
//     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filef(R_path + "/R/Our/data/Test_" + 
//                 TestNumber + "/alpha_" + alpha_string + "/n_" + seq_n_string[n] + "/fnCpp_" + alpha_string + ".csv");

//     if (filef.is_open()){
//       filef << computedF.format(CSVFormatf);
//       filef.close();
//     }

//     DVector<double> computedBeta = model.beta();
//     const static Eigen::IOFormat CSVFormat_beta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream file_beta(R_path + "/R/Our/data/Test_" + 
//                 TestNumber + "/alpha_" + alpha_string + "/n_" + seq_n_string[n] + "/betaCpp_" + alpha_string + ".csv");
//     if (file_beta.is_open()){
//       file_beta << computedBeta.format(CSVFormat_beta);
//       file_beta.close();
//     }


//     double J = model.J_final_sqrpde();
//     std::ofstream fileJ(R_path + "/R/Our/data/Test_" + 
//                 TestNumber + "/alpha_" + alpha_string + "/n_" + seq_n_string[n] + "/JCpp.csv");
//     if (fileJ.is_open()){
//       fileJ << J;
//       fileJ.close();
//     }

//     std::size_t niter = model.niter_sqrpde();
//     std::ofstream filen(R_path + "/R/Our/data/Test_" + 
//                 TestNumber + "/alpha_" + alpha_string + "/n_" + seq_n_string[n] + "/niterCpp.csv");
//     if (filen.is_open()){
//       filen << niter;
//       filen.close();
//     }

//   }
  
// }


/* test 5
   domain:       unit square
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1

 */

TEST(SQRPDE, Test5_Laplacian_SemiParametric_GeostatisticalAtNodes) {
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  double alpha = 0.5; 
  const std::string alpha_string = "50"; 
  const std::string TestNumber = "5"; 

  SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);

  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  CSVFile<double> XFile; // covariates file
  // yFile = reader.parseFile("data/models/SQRPDE/2D_test" + TestNumber + "/z.csv");
  std::string data_macro_strategy_type = "skewed_data"; 
  std::string data_strategy_type = "E"; 

  // Marco
  std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
  // Ilenia 
  // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
  yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
                  TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
                  "/z.csv");             
  DMatrix<double> y = yFile.toEigen();

  XFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
                  TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
                  "/X.csv");             
  DMatrix<double> X = XFile.toEigen();


  // set model data
  BlockFrame<double, int> df;
  df.insert(OBSERVATIONS_BLK,  y);
  df.insert(DESIGN_MATRIX_BLK, X);
  model.setData(df);

  std::vector<double> seq_tol_weights = {0.00000001, 0.0000001, 0.000001}; 
  std::vector<std::string> seq_tol_weights_string = {"1e-08", "1e-07", "1e-06"}; 

  std::vector<double> seq_tol_FPIRLS = {0.000000001, 0.00000001, 0.0000001, 0.000001};
  std::vector<std::string> seq_tol_FPIRLS_string = {"1e-09", "1e-08", "1e-07", "1e-06"}; 

  std::string lin_sys_solver = "LU";    // depends on the "symmetry" option in R 

  CSVFile<double> lambdaCSV; 
  DMatrix<double> lambda; 

  for(int i = 0; i < seq_tol_weights.size(); ++i ){
    for(int j = 0; j < seq_tol_FPIRLS.size(); ++j){

      // use optimal lambda to avoid possible numerical issues
      lambdaCSV = reader.parseFile(R_path + "/R/Our/data/Test_" + 
                  TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
                  "/" + lin_sys_solver + "/LambdaR_" + alpha_string + ".csv");     // from R 
      
      lambda = lambdaCSV.toEigen();
      model.setLambdaS(lambda(0,0));

      // solve smoothing problem
      model.setTolerances(seq_tol_weights[i], seq_tol_FPIRLS[j]); 
      model.init();     
      model.solve();

      // Save C++ solution 
      DMatrix<double> computedF = model.f();
      const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
      // std::ofstream filef("data/models/SQRPDE/2D_test1/fnCpp_" + alpha_string + ".csv");
      std::ofstream filef(R_path + "/R/Our/data/Test_" 
                  + TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
                  "/" + lin_sys_solver + "/fnCpp_" + alpha_string + ".csv");

      if (filef.is_open()){
        filef << computedF.format(CSVFormatf);
        filef.close();
      }


      DVector<double> computedBeta = model.beta();
      const static Eigen::IOFormat CSVFormat_beta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
      std::ofstream file_beta(R_path + "/R/Our/data/Test_" 
                  + TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
                  "/" + lin_sys_solver + "/betaCpp_" + alpha_string + ".csv");
      if (file_beta.is_open()){
        file_beta << computedBeta.format(CSVFormat_beta);
        file_beta.close();
      }

      double J = model.J_final_sqrpde();
      std::ofstream fileJ(R_path + "/R/Our/data/Test_" 
                  + TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
                  "/" + lin_sys_solver + "/JCpp.csv");
      if (fileJ.is_open()){
        fileJ << J;
        fileJ.close();
      }

      std::size_t niter = model.niter_sqrpde();
      std::ofstream filen(R_path + "/R/Our/data/Test_" 
                  + TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
                  "/" + lin_sys_solver + "/niterCpp.csv");
      if (filen.is_open()){
        filen << niter;
        filen.close();
      }


    }
  }


}


/* test 6
   domain:       horseshoe_medium
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   Data generation: Norm(mu,sigma) with (mu,sigma) generated through fs.test

 */

TEST(SQRPDE, Test6_Laplacian_NonParametric_GeostatisticalAtLocations) {
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("c_shaped_medium");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  double alpha = 0.1; 
  const std::string alpha_string = "10"; 
  const std::string TestNumber = "6"; 

  SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);

  // Marco
  // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
  // Ilenia 
  std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  

  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + TestNumber + "/alpha_" + alpha_string + "/z.csv");             
  DMatrix<double> y = yFile.toEigen();

  // load locations where data are sampled
  CSVFile<double> locFile;
  locFile = reader.parseFile(R_path + "/R/Our/data/Test_" + TestNumber + "/alpha_" + alpha_string + "/locs.csv");
  DMatrix<double> loc = locFile.toEigen();

  model.set_spatial_locations(loc);

  // set model data
  BlockFrame<double, int> df;
  df.insert(OBSERVATIONS_BLK,  y);
  model.setData(df);

  std::vector<double> seq_tol_weights = {0.00000001, 0.000001}; 
  std::vector<std::string> seq_tol_weights_string = {"1e-08",  "1e-06"}; 

  std::vector<double> seq_tol_FPIRLS = {0.000000001, 0.00000001, 0.0000001, 0.000001};
  std::vector<std::string> seq_tol_FPIRLS_string = {"1e-09", "1e-08", "1e-07", "1e-06"}; 

  CSVFile<double> lambdaCSV; 
  DMatrix<double> lambda; 

  for(int i = 0; i < seq_tol_weights.size(); ++i ){
    for(int j = 0; j < seq_tol_FPIRLS.size(); ++j){

      // use optimal lambda to avoid possible numerical issues
      lambdaCSV = reader.parseFile(R_path + "/R/Our/data/Test_" + 
                  TestNumber + "/alpha_" + alpha_string + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
                  "/GCV/Exact/LambdaR_" + alpha_string + ".csv");     // from R 
      
      lambda = lambdaCSV.toEigen();
      model.setLambdaS(lambda(0,0));

      // solve smoothing problem
      model.setTolerances(seq_tol_weights[i], seq_tol_FPIRLS[j]); 
      model.init();     
      model.solve();

      // Save C++ solution 
      DMatrix<double> computedF = model.Psi()*model.f();
      const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
      // std::ofstream filef("data/models/SQRPDE/2D_test1/fnCpp_" + alpha_string + ".csv");
      std::ofstream filef(R_path + "/R/Our/data/Test_" 
                  + TestNumber + "/alpha_" + alpha_string + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
                  "/fnCpp_" + alpha_string + ".csv");

      if (filef.is_open()){
        filef << computedF.format(CSVFormatf);
        filef.close();
      }

      // Matrix of weights at convergence 
      DMatrix<double> W_conv = model.W(); 
      const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
      std::ofstream file_W(R_path + "/R/Our/data/Test_" 
                  + TestNumber + "/alpha_" + alpha_string + "/"  + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
                  "/WCpp_" + alpha_string + ".csv");
      if(file_W.is_open()){
        file_W << W_conv.format(CSVFormatW) << '\n' ; 
      }


      double J = model.J_final_sqrpde();
      std::ofstream fileJ(R_path + "/R/Our/data/Test_" 
                  + TestNumber + "/alpha_" + alpha_string + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + "/JCpp.csv");
      if (fileJ.is_open()){
        fileJ << J;
        fileJ.close();
      }

      std::size_t niter = model.niter_sqrpde();
      std::ofstream filen(R_path + "/R/Our/data/Test_" 
                  + TestNumber + "/alpha_" + alpha_string + 
                  "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + "/niterCpp.csv");
      if (filen.is_open()){
        filen << niter;
        filen.close();
      }


    }
  }


}





// Debug
/*   **  test correctness of computed results  **   */

  // // Initial mu (the vector returned by initialize_mu)  
  // DVector<double> computedInit = model.get_mu_init();
  // const static Eigen::IOFormat CSVFormatInit(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream fileInit("data/models/SQRPDE/2D_test2/mu_initCpp_" + alpha_string + ".csv");
  // if (fileInit.is_open()){
  //   fileInit << computedInit.format(CSVFormatInit);
  //   fileInit.close();
  // }

  
  // // Matrix of pseudo observations 
  // DMatrix<double> computed_matrix_pseudo = model.get_matrix_pseudo(); 
  // const static Eigen::IOFormat CSVFormatpseudo(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream file_pseudo("data/models/SQRPDE/2D_test2/matrix_pseudoCpp_" + alpha_string + ".csv");
  // if(file_pseudo.is_open()){
  //   file_pseudo << computed_matrix_pseudo.format(CSVFormatpseudo) << '\n' ; 
  // }

  // // Matrix of weights
  // DMatrix<double> computed_matrix_weights = model.get_matrix_weight(); 
  // const static Eigen::IOFormat CSVFormatweights(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream file_weights("data/models/SQRPDE/2D_test2/matrix_weightsCpp_" + alpha_string + ".csv");
  // if(file_weights.is_open()){
  //   file_weights << computed_matrix_weights.format(CSVFormatweights) << '\n' ; 
  // }

  // // Matrix of weights at convergence 
  // DMatrix<double> W_conv = model.W(); 
  // const static Eigen::IOFormat CSVFormatW(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream file_W("data/models/SQRPDE/2D_test2/matrix_WCpp_" + alpha_string + ".csv");
  // if(file_W.is_open()){
  //   file_W << W_conv.format(CSVFormatW) << '\n' ; 
  // }


  // // Matrix of abs_res
  // DMatrix<double> computed_abs_res = model.get_matrix_abs_res(); 
  // const static Eigen::IOFormat CSVFormatabsres(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream file_absres("data/models/SQRPDE/2D_test2/matrix_absresCpp_" + alpha_string + ".csv");
  // if(file_absres.is_open()){
  //   file_absres << computed_abs_res.format(CSVFormatabsres) << '\n' ; 
  // }


  // // Matrix of beta
  // DMatrix<double> computed_allbeta = model.get_matrix_beta() ; 
  // const static Eigen::IOFormat CSVFormatAllBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream file_allBeta("data/models/SQRPDE/2D_test2/matrix_betaCpp_" + alpha_string + ".csv");
  // if(file_allBeta.is_open()){
  //   file_allBeta << computed_allbeta.format(CSVFormatAllBeta) << '\n' ; 
  // }