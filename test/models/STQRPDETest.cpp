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

#include "../../fdaPDE/preprocess/InitialConditionEstimator.h"
using fdaPDE::preprocess::InitialConditionEstimator;

#include<fstream>
#include<iostream>



/* test 1 
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
TEST(SQRPDE_Time, Test1_Laplacian_NonParametric_GeostatisticalAtNodes_Separable_Monolithic) {

  // Parameters 
  const std::string TestNumber = "1"; 

  std::vector<double> alphas = {0.1, 0.5, 0.9}; 

  // number of simulations
  unsigned int n_sim = 6;
  
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
  double lambdaS ; // smoothing in space
  double lambdaT ; // smoothing in time

  for(double alpha : alphas){

    unsigned int alpha_int = alpha*100; 

    std::cout << "------------------------------------------alpha=" << std::to_string(alpha_int) << "%-------------------------------------------------" << std::endl; 

    SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatMeshNodes,
      fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);


    for(unsigned int sim = 1; sim <= n_sim; ++sim){
    
    // // Read lambda
    // std::ifstream fileLambdaS(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/LambdaS.csv");
    // if (fileLambdaS.is_open()){
    //   fileLambdaS >> lambdaS; 
    //   fileLambdaS.close();
    // }

    // std::ifstream fileLambdaT(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/LambdaT.csv");
    // if (fileLambdaT.is_open()){
    //   fileLambdaT >> lambdaT; 
    //   fileLambdaT.close();
    // }

    lambdaS = std::pow(10,-4.4);
    lambdaT = std::pow(10,-4.4);

    model.setLambdaS(lambdaS);
    model.setLambdaT(lambdaT);

    std::cout << "LamnbdaS = " << lambdaS << std::endl ; 
    std::cout << "LamnbdaT = " << lambdaT << std::endl ; 


      // load data from .csv files
      CSVReader<double> reader{};
      CSVFile<double> yFile; // observation file
      yFile = reader.parseFile(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/y.csv");
      DMatrix<double> y = yFile.toEigen();

      // set model data
      BlockFrame<double, int> df;
      df.stack(OBSERVATIONS_BLK, y);
      model.setData(df);
      
      // // solve smoothing problem
      model.init();
      model.solve();

      // Save C++ solution 
      DMatrix<double> computedF = model.f();
      const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
      // std::ofstream filef(path_test + "/alpha_" + alpha_string + "/fCpp.csv");
      std::ofstream filef(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/fCpp.csv");
      if (filef.is_open()){
            filef << computedF.format(CSVFormatf);
            filef.close();
      }

      // Save n_iter
      std::ofstream filen_iter(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/n_iter.csv");
        if (filen_iter.is_open()){
          filen_iter << std::setprecision(16) << model.n_iter();
          filen_iter.close();
        }

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
TEST(SQRPDE_Time, Test2_Laplacian_SemiParametric_GeostatisticalAtLocations_Separable_Monolithic) {

  // Parameters 
  const std::string TestNumber = "2"; 

  std::vector<double> alphas = {0.1, 0.5, 0.9};

  // number of simulations
  unsigned int n_sim = 10;
  
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

  // define statistical model
  double lambdaS ; // smoothing in space
  double lambdaT ; // smoothing in time

  // load sample position and covariates
  CSVReader<double> reader{};
  CSVFile<double> locFile; // locations file
  locFile = reader.parseFile(path_test + "/locs.csv");
  DMatrix<double> loc = locFile.toEigen();

  CSVFile<double> XFile; // design matrix
  XFile = reader.parseFile  (path_test + "/X.csv");
  DMatrix<double> X = XFile.toEigen();


  for(double alpha : alphas){

    unsigned int alpha_int = alpha*100; 

    std::cout << "------------------------------------------alpha=" << std::to_string(alpha_int) << "%-------------------------------------------------" << std::endl; 

    SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatLocations,
  	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);

    model.set_spatial_locations(loc);


    for(unsigned int sim = 1; sim <= n_sim; ++sim){
    
      // Read lambda
      std::ifstream fileLambdaS(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/LambdaS.csv");
      if (fileLambdaS.is_open()){
        fileLambdaS >> lambdaS; 
        fileLambdaS.close();
      }

      std::ifstream fileLambdaT(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/GCV/Exact/LambdaT.csv");
      if (fileLambdaT.is_open()){
        fileLambdaT >> lambdaT; 
        fileLambdaT.close();
      }

      model.setLambdaS(lambdaS);
      model.setLambdaT(lambdaT);

      // load data from .csv files
      CSVFile<double> yFile; // observation file
      yFile = reader.parseFile  (path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/y.csv");
      DMatrix<double> y = yFile.toEigen();

      // set model data
      BlockFrame<double, int> df;
      df.stack (OBSERVATIONS_BLK,  y);
      df.stack (DESIGN_MATRIX_BLK, X);
      model.setData(df);
      
      // solve smoothing problem
      model.init();
      model.solve();

      
      // Save C++ solution 
      DMatrix<double> computedF = model.f();
      const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
      std::ofstream filef(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/fCpp.csv");
      if (filef.is_open()){
            filef << computedF.format(CSVFormatf);
            filef.close();
      }

      
      // Save C++ solution 
      DMatrix<double> computedFn = model.Psi()*model.f();
      const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
      std::ofstream filefn(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/fnCpp.csv");
      if (filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
      }


      // Save beta
      DMatrix<double> computedbeta = model.beta();
      const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
      std::ofstream filebeta(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/betaCpp.csv");
      if (filebeta.is_open()){
            filebeta << computedbeta.format(CSVFormatbeta);
            filebeta.close();
      } 

      // Save n_iter
      std::ofstream filen_iter(path_test + "/alpha_" + std::to_string(alpha_int) + "/sim_" + std::to_string(sim) + "/n_iter.csv");
        if (filen_iter.is_open()){
          filen_iter << std::setprecision(16) << model.n_iter();
          filen_iter.close();
        }

    }

  }
  
}






// /* test 3
//    domain:       quasicircular domain
//    sampling:     areal
//    penalization: non-costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
//    time penalization: parabolic (monolithic solution)
//  */
// TEST(SQTRPDE, Test3_NonCostantCoefficientsPDE_NonParametric_Areal_Parabolic_Monolithic_EstimatedIC) {
//   // define time domain, we skip the first time instant because we are going to use the first block of data
//   // for the estimation of the initial condition
//   DVector<double> time_mesh;
//   time_mesh.resize(11);
//   for(std::size_t i = 0; i < 10; ++i) time_mesh[i] = 0.4*i;

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("quasi_circle");
//   // load PDE coefficients data
//   CSVReader<double> reader{};
//   CSVFile<double> diffFile; // diffusion tensor
//   diffFile = reader.parseFile("data/models/STQRPDE/2D_test3/K.csv");
//   DMatrix<double> diffData = diffFile.toEigen();
//   CSVFile<double> adveFile; // transport vector
//   adveFile = reader.parseFile("data/models/STQRPDE/2D_test3/b.csv");
//   DMatrix<double> adveData = adveFile.toEigen();

//   // define non-constant coefficients
//   SpaceVaryingDiffusion<2> diffCoeff;
//   diffCoeff.setData(diffData);
//   SpaceVaryingAdvection<2> adveCoeff;
//   adveCoeff.setData(adveData);
//   // parabolic PDE
//   auto L = dT() + Laplacian(diffCoeff.asParameter()) + Gradient(adveCoeff.asParameter());
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, time_mesh.rows()); 
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
//   // define statistical model
//   double lambdaS = std::pow(0.1, 8); // smoothing in space
//   double lambdaT = std::pow(0.1, 8); // smoothing in time

//   CSVReader<int> int_reader{};
//   CSVFile<int> arealFile; // incidence matrix for specification of subdomains
//   arealFile = int_reader.parseFile("data/models/STQRPDE/2D_test3/incidence_matrix.csv");
//   DMatrix<int> areal = arealFile.toEigen();
  
//   double alpha = 0.5; 
//   SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeParabolic, fdaPDE::models::Areal,
// 	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);
//   model.setLambdaS(lambdaS);
//   model.setLambdaT(lambdaT);
//   model.set_spatial_locations(areal);

//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/STQRPDE/2D_test3/y.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.stack (OBSERVATIONS_BLK, y);
//   model.setData(df);
  
//   // define initial condition estimator over grid of lambdas
//   InitialConditionEstimator ICestimator(model);
//   std::vector<SVector<1>> lambdas;
//   for(double x = -9; x <= 3; x += 0.1) lambdas.push_back(SVector<1>(std::pow(10,x))); 
//   // compute estimate
//   ICestimator.apply(lambdas);
//   DMatrix<double> ICestimate = ICestimator.get();
//   // // test computation initial condition
//   // CSVFile<double> ICfile;
//   // ICfile = reader.parseFile("data/models/STQRPDE/2D_test3/IC.csv");  
//   // DMatrix<double> expectedIC = ICfile.toEigen();
//   // EXPECT_TRUE( almost_equal(expectedIC, ICestimate) );

//   // set estimated initial condition
//   model.setInitialCondition(ICestimate);
//   model.shift_time(1); // shift data one time instant forward
  
//   model.init();
//   model.solve();
  
//   // //   **  test correctness of computed results  **   

//   // DMatrix<double> computedF;
//   // computedF.resize((model.n_temporal_locs()+1)*model.n_basis(), 1);
//   // computedF << model.s(), model.f();
  
//   // // estimate of spatial field \hat f (with estimatate of initial condition)
//   // SpMatrix<double> expectedSolution;
//   // Eigen::loadMarket(expectedSolution, "data/models/STQRPDE/2D_test3/sol.mtx");
//   // std::size_t N = computedF.rows();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );

//   const std::string path_test = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/STQRPDE/2D_test3"; 
//   // Save C++ solution 
//   DMatrix<double> computedF = model.f();
//   const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   std::ofstream filef(path_test + "/f.csv");
//   if(filef.is_open()){
//     filef << computedF.format(CSVFormatf);
//     filef.close();
//   }

//   DMatrix<double> computedFn = model.Psi()*computedF;
//   const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   std::ofstream filefn(path_test + "/fn.csv");
//   if(filefn.is_open()){
//     filefn << computedFn.format(CSVFormatfn);
//     filefn.close();
//   }


// }

/* test 4
   domain:       unit square 
   sampling:     locations != nodes 
   penalization: laplacian 
   covariates:   yes
   BC:           no
   order FE:     1
   time penalization: parabolic
 */
// TEST(STQRPDE, Test4_Laplacian_SemiParametric_Locations_Parabolic_Monolithic_EstimatedIC){

//   // Solver type
//   std::string solver = "Iterative";   // Iterative 

//   // define time domain, we skip the first time instant because we are going to use the first block of data
//   // for the estimation of the initial condition
//   DVector<double> time_mesh;
//   time_mesh.resize(6);
//   for(std::size_t i = 0; i < 5; ++i) time_mesh[i] = 0.4*i;

// //   // define domain and regularizing PDE
// //   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
  
// //   // parabolic PDE
// //   auto L = dT() + Laplacian();
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, time_mesh.rows()); 
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
//   // define statistical model
//   double lambdaS = std::pow(0.1, 6); // smoothing in space
//   double lambdaT = std::pow(0.1, 6); // smoothing in time

//   std::string lambda_string; 
//   if(almost_equal(lambdaS, 1e-06) && almost_equal(lambdaT, 1e-06))
//     lambda_string = "1e-06"; 

//   std::vector<double> alphas = {0.1, 0.5, 0.9};

//   // load sample position
//   CSVReader<double> reader{};
//   CSVFile<double> locFile; // locations file
//   locFile = reader.parseFile("/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/space_time/Test_4/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile("/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/space_time/Test_4/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // number of simulations
//   unsigned int n_sim = 10; 

//   for(double alpha : alphas){

//     unsigned int alpha_int = alpha*100; 

//     std::cout << "------------------------------------------alpha=" << std::to_string(alpha_int) << "%-------------------------------------------------" << std::endl; 

//     for(unsigned int sim = 1; sim <= n_sim; ++sim){

//       const std::string path_test = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/space_time/Test_4/results_lambdas_" + lambda_string + "/alpha_" + std::to_string(alpha_int) + "/" + solver + "/sim_" + std::to_string(sim);

//       // load data from .csv files
//       CSVFile<double> yFile; // observation file
//       yFile = reader.parseFile(path_test + "/y.csv");
//       DMatrix<double> y = yFile.toEigen();

//       BlockFrame<double, int> df;
//       df.stack(OBSERVATIONS_BLK, y);
//       df.stack(DESIGN_MATRIX_BLK, X);

//       SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeParabolic, fdaPDE::models::GeoStatLocations,
//             fdaPDE::models::IterativeSolver> model(problem, time_mesh, alpha);
//       model.setLambdaS(lambdaS);
//       model.setLambdaT(lambdaT);
//       model.set_spatial_locations(loc);
//       model.setData(df);

//       // define initial condition estimator over grid of lambdas
//       InitialConditionEstimator ICestimator(model);
//       std::vector<SVector<1>> lambdas_IC;
//       for(double x = -9; x <= 3; x += 0.1) lambdas_IC.push_back(SVector<1>(std::pow(10,x))); 
//       // compute estimate
//       ICestimator.apply(lambdas_IC);
//       DMatrix<double> ICestimate = ICestimator.get();

//       // set estimated initial condition
//       model.setInitialCondition(ICestimate);
//       model.shift_time(1); // shift data one time instant forward
      
//       model.init();
//       model.solve();
  
//       // Save C++ solution 
//       DMatrix<double> computedF = model.f();
//       const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//       std::ofstream filef(path_test + "/f.csv");
//       if(filef.is_open()){
//         filef << computedF.format(CSVFormatf);
//         filef.close();
//       }

//       DMatrix<double> computedFn = model.Psi()*computedF;
//       const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//       std::ofstream filefn(path_test + "/fn.csv");
//       if(filefn.is_open()){
//         filefn << computedFn.format(CSVFormatfn);
//         filefn.close();
//       }

//       DMatrix<double> computedBeta = model.beta();
//       const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//       std::ofstream filebeta(path_test + "/beta.csv");
//       if(filebeta.is_open()){
//         filebeta << computedBeta.format(CSVFormatbeta);
//         filebeta.close();
//       }

//       unsigned int computedNiter = model.n_iter();
//       std::ofstream ofile_niter;
//       ofile_niter.open(path_test + "/n_iter.txt");
//       ofile_niter << computedNiter; 
//       ofile_niter.close(); 

//     }

//   }
// }