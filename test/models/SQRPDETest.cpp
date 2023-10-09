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


// /* test 1
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//  */
// TEST(SQRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u);  // definition of regularizing PDE

//   // define statistical model
//   // use optimal lambda to avoid possible numerical issues
//   double lambda = 1.778279*std::pow(0.1, 4);
//   double alpha = 0.1; 
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);
//   model.setLambdaS(lambda);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test1/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);

//   // solve smoothing problem
//   model.init();
//   model.solve();

//   /*   **  test correctness of computed results  **   */
//   // estimate of spatial field \hat f
//   SpMatrix<double> expectedSolution;
//   Eigen::loadMarket(expectedSolution,   "data/models/SQRPDE/2D_test1/sol.mtx");
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
  
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );

//   // Eigen::saveMarket(computedF, "data/models/SQRPDE/2D_test1/sol.mtx"); 

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
//   MeshLoader<Mesh2D<>> domain("c_shaped");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   CSVReader<double> reader{};
//   // load locations where data are sampled
//   CSVFile<double> locFile;
//   locFile = reader.parseFile("data/models/SQRPDE/2D_test2/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   // use optimal lambda to avoid possible numerical issues
//   double alpha = 0.9;
//   double lambda = 3.162277660168379*std::pow(0.1, 4);
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);
//   model.setLambdaS(lambda);
//   model.set_spatial_locations(loc);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test2/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile("data/models/SQRPDE/2D_test2/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();

//   /*   **  test correctness of computed results  **   */

//   // estimate of spatial field \hat f
//   SpMatrix<double> expectedSolution;
//   Eigen::loadMarket(expectedSolution, "data/models/SQRPDE/2D_test2/sol.mtx");
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );


//   // estimate of coefficient vector \hat \beta
//   SpMatrix<double> expectedBeta;
//   Eigen::loadMarket(expectedBeta, "data/models/SQRPDE/2D_test2/beta.mtx");
//   DVector<double> computedBeta = model.beta();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedBeta), computedBeta) );

// }


// /* test 3
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
//  */
// TEST(SQRPDE, Test3_CostantCoefficientsPDE_NonParametric_GeostatisticalAtNodes) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");

//   // non unitary diffusion tensor
//   SMatrix<2> K;
//   K << 1,0,0,4;
//   auto L = Laplacian(K); // anisotropic diffusion
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double lambda = 5.623413251903491*pow(0.1,4);
//   double alpha = 0.1; 
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);
//   model.setLambdaS(lambda);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test3/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);

//   // solve smoothing problem
//   model.init();
//   model.solve();

//   /*   **  test correctness of computed results  **   */
//   // estimate of spatial field \hat f
//   SpMatrix<double> expectedSolution;
//   Eigen::loadMarket(expectedSolution,   "data/models/SQRPDE/2D_test3/sol.mtx");
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
  
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );

// }


// /* test 4
//    domain:       c-shaped
//    sampling:     areal
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//  */
// TEST(SQRPDE, Test4_Laplacian_SemiParametric_Areal) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped_areal");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   CSVReader<int> int_reader{};
//   CSVFile<int> arealFile; // incidence matrix for specification of subdomains
//   arealFile = int_reader.parseFile("data/models/SQRPDE/2D_test4/incidence_matrix.csv");
//   DMatrix<int> areal = arealFile.toEigen();

// // use optimal lambda to avoid possible numerical issues
//   double alpha = 0.5;
//   double lambda = 5.623413251903491*std::pow(0.1, 3);
//   SQRPDE<decltype(problem), fdaPDE::models::Areal> model(problem, alpha);
//   model.setLambdaS(lambda);
//   model.set_spatial_locations(areal);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test4/z.csv");

//   DMatrix<double> y = yFile.toEigen();

//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile("data/models/SQRPDE/2D_test4/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();


//   /*   **  test correctness of computed results  **   */

//   // estimate of spatial field \hat f
//   SpMatrix<double> expectedSolution;
//   Eigen::loadMarket(expectedSolution, "data/models/SQRPDE/2D_test4/sol.mtx");
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );


//   // estimate of coefficient vector \hat \beta
//   SpMatrix<double> expectedBeta;
//   Eigen::loadMarket(expectedBeta, "data/models/SQRPDE/2D_test4/beta.mtx");
//   DVector<double> computedBeta = model.beta();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedBeta), computedBeta) ); 

// }


// -------- Test presi da STRPDE

/* test 1
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
// TEST(SQTRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes_Separable_Monolithic) {
//   // define time domain
//   DVector<double> time_mesh;
//   time_mesh.resize(11);
//   std::size_t i = 0;
//   for(double x = 0; x <= 2; x+=0.2, ++i) time_mesh[i] = x;

//   // time_mesh ha lunghezza 11 ed anche il vettore time_ della classe
//   // però basis_.size()=13 
  
//   // define spatial domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3*time_mesh.rows(), 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  
//   // define statistical model
//   double alpha = 0.5;
//   double lambdaS = 0.01; // smoothing in space
//   double lambdaT = 0.01; // smoothing in time
//   SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatMeshNodes,
// 	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);
//   model.setLambdaS(lambdaS);
//   model.setLambdaT(lambdaT);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/STRPDE/2D_test1/y.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.stack(OBSERVATIONS_BLK, y);
//   model.setData(df);
  
//   // // solve smoothing problem
//   model.init();
//   model.solve();

//   // //    **  test correctness of computed results  **   
  
//   // // // \Psi matrix
//   // SpMatrix<double> expectedPsi;
//   // Eigen::loadMarket(expectedPsi, "data/models/STRPDE/2D_test1/Psi.mtx");
//   // SpMatrix<double> computedPsi = model.Psi();
//   // EXPECT_TRUE( almost_equal(expectedPsi, computedPsi) );

//   // // R0 matrix (discretization of identity operator)
//   // SpMatrix<double> expectedR0;
//   // Eigen::loadMarket(expectedR0,  "data/models/STRPDE/2D_test1/R0.mtx");
//   // SpMatrix<double> computedR0 = model.R0();
//   // EXPECT_TRUE( almost_equal(expectedR0, computedR0) );
  
//   // // R1 matrix (discretization of differential operator)
//   // SpMatrix<double> expectedR1;
//   // Eigen::loadMarket(expectedR1,  "data/models/STRPDE/2D_test1/R1.mtx");
//   // SpMatrix<double> computedR1 = model.R1();
//   // EXPECT_TRUE( almost_equal(expectedR1, computedR1) );
  
//   // // estimate of spatial field \hat f
//   // SpMatrix<double> expectedSolution;
//   // Eigen::loadMarket(expectedSolution,   "data/models/STRPDE/2D_test1/sol.mtx");
//   // DMatrix<double> computedF = model.f();
//   // std::size_t N = computedF.rows();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );
// }

// alpha = 50% -> è meglio con i dati
// con la nostra inizializzazione specifica : 23 iter
// restituendo i dati :                       13 iter

// alpha = 10% -> eqivalenti
// con la nostra inizializzazione specifica : 83 iter, J = 0.0480375
// restituendo i dati :                       86 iter, J = 0.0480371

// alpha = 1% -> eqivalenti
// con la nostra inizializzazione specifica : 98 iter, J = 0.0018945
// restituendo i dati :                       94 iter, J = 0.00189547

// alpha = 99% -> equivalenti
// con la nostra inizializzazione specifica : 115 iter, J = 0.001132
// restituendo i dati :                       118 iter, J = 0.00113163



// /* test 2
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    time penalization: separable (mass penalization)
//  */
// TEST(STQRPDE, Test2_Laplacian_SemiParametric_GeostatisticalAtLocations_Separable_Monolithic) {
//   // define time domain
//   DVector<double> time_mesh;
//   time_mesh.resize(5);
//   for(std::size_t i = 0; i < 5; ++i)
//     time_mesh[i] = (fdaPDE::testing::pi/4)*i;

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
//   // define statistical model
//   double lambdaS = 0.01; // smoothing in space
//   double lambdaT = 0.01; // smoothing in time
//   // load sample position
//   CSVReader<double> reader{};
//   CSVFile<double> locFile; // locations file
//   locFile = reader.parseFile("data/models/STRPDE/2D_test2/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   // Define model 
//   double alpha = 0.1;
//   SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatLocations,
// 	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);
//   model.setLambdaS(lambdaS);
//   model.setLambdaT(lambdaT);
//   model.set_spatial_locations(loc);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/STRPDE/2D_test2/y.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile  ("data/models/STRPDE/2D_test2/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.stack (OBSERVATIONS_BLK,  y);
//   df.stack (DESIGN_MATRIX_BLK, X);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();

//   //   **  test correctness of computed results  **   
  
//   // // \Psi matrix (sensible to locations != nodes)
//   // SpMatrix<double> expectedPsi;
//   // Eigen::loadMarket(expectedPsi, "data/models/STRPDE/2D_test2/Psi.mtx");
//   // SpMatrix<double> computedPsi = model.Psi();
//   // EXPECT_TRUE( almost_equal(expectedPsi, computedPsi) );

//   // // R0 matrix (discretization of identity operator)
//   // SpMatrix<double> expectedR0;
//   // Eigen::loadMarket(expectedR0,  "data/models/STRPDE/2D_test2/R0.mtx");
//   // SpMatrix<double> computedR0 = model.R0();
//   // EXPECT_TRUE( almost_equal(expectedR0, computedR0) );
  
//   // // R1 matrix (discretization of differential operator)
//   // SpMatrix<double> expectedR1;
//   // Eigen::loadMarket(expectedR1,  "data/models/STRPDE/2D_test2/R1.mtx");
//   // SpMatrix<double> computedR1 = model.R1();
//   // EXPECT_TRUE( almost_equal(expectedR1, computedR1) );

//   // // estimate of spatial field \hat f
//   // SpMatrix<double> expectedSolution;
//   // Eigen::loadMarket(expectedSolution, "data/models/STRPDE/2D_test2/sol.mtx");
//   // DMatrix<double> computedF = model.f();
//   // std::size_t N = computedF.rows();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution), computedF) );

//   // // estimate of coefficient vector \hat \beta
//   // SpMatrix<double> expectedBeta;
//   // Eigen::loadMarket(expectedBeta, "data/models/STRPDE/2D_test2/beta.mtx");
//   DVector<double> computedBeta = model.beta();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedBeta), computedBeta) );
  
//   for(std::size_t i = 0; i < computedBeta.size(); ++i)
//     std::cout << computedBeta[i] << std::endl ;   

// }
// alpha = 50% -> meglio con i dati
// con la nostra inizializzazione specifica : 45 iter, beta_hat = 2.00679  (da considerare che la nostra iniz pone beta=0)
// inizializzando con i dati :                19 iter, beta_hat = 2.00707

// alpha = 1% -> un po' meglio i dati
// con la nostra inizializzazione specifica : 127 iter, beta_hat = 2.04351, J = 0.0131223
// inizializzando con i dati :                134 iter, beta_hat = 2.04351, J = 0.0131225


// alpha = 10% -> meglio la nostra  -> UNICO CASO ! 
// con la nostra inizializzazione specifica : 110 iter, beta_hat = 2.01316, J = 0.0972953  
// inizializzando con i dati :                178 iter, beta_hat = 2.01313, J = 0.0972994

// alpha = 99% -> equivalenti
// con la nostra inizializzazione specifica : 200 iter, beta_hat = 2.07865, J = 0.0161182
// inizializzando con i dati :                200 iter, beta_hat = 2.07864, J = 0.0161226




/* test 5
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes, time locations != time nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
// TEST(STRPDE, Test5_Laplacian_NonParametric_GeostatisticalAtNodes_TimeLocations_Separable_Monolithic) {
//   // define time domain
//   DVector<double> time_mesh;
//   time_mesh.resize(11);
//   std::size_t i = 0;
//   for(double x = 0; x <= 2; x+=0.2, ++i) time_mesh[i] = x;
  
//   // define spatial domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3*time_mesh.rows(), 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> timeLocationsFile;
//   timeLocationsFile = reader.parseFile("data/models/STRPDE/2D_test5/time_locations.csv");
//   DMatrix<double> timeLocations = timeLocationsFile.toEigen();
  
//   // define statistical model
//   double alpha = 0.99; 
//   double lambdaS = 0.01; // smoothing in space
//   double lambdaT = 0.01; // smoothing in time
//   SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatMeshNodes,
// 	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);
//   model.setLambdaS(lambdaS);
//   model.setLambdaT(lambdaT);
//   model.set_temporal_locations(timeLocations);
  
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/STRPDE/2D_test5/y.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.stack(OBSERVATIONS_BLK, y);
//   model.setData(df);
  
//   // // solve smoothing problem
//   model.init();
//   model.solve();

//   //    **  test correctness of computed results  **   
  
//   // // estimate of spatial field \hat f
//   // SpMatrix<double> expectedSolution;
//   // Eigen::loadMarket(expectedSolution,   "data/models/STRPDE/2D_test5/sol.mtx");
//   // DMatrix<double> computedF = model.f();
//   // std::size_t N = computedF.rows();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );
// }

// alpha = 1% -> equivalenti
// con la nostra inizializzazione specifica :  90 iter, J = 0.00160683
// inizializzando con i dati :                 92 iter, J = 0.0016071

// alpha = 10% -> non ci sono diff significative
// con la nostra inizializzazione specifica :  iter 119, J = 0.0467747
// inizializzando con i dati : 122 iter , J = 0.0467749

// alpha = 99% -> equivalenti
// con la nostra inizializzazione specifica :  82 iter, J = 0.00129562
// inizializzando con i dati :                 82 iter, J = 0.00129555


// Da questi test (ovvero i test di STRPDE) sembra meglio l'inizializzazione con i dati 




// ---- Test con dati generati da script R 

/* test 1 
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
// TEST(SQTRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes_Separable_Monolithic) {

//   // Parameters 
//   const std::string TestNumber = "1"; 

//   double alpha = 0.50;
//   unsigned int alpha_int = alpha*100; 
//   const std::string alpha_string = std::to_string(alpha_int);

//   // const std::string init_type = "init_dati"; 
  
//   // Marco
//   // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
//   // Ilenia 
//   std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared"; 

//   std::string path_test = path + "/space_time/Test_" + TestNumber ;


//   // define time domain
//   DVector<double> time_mesh;
//   time_mesh.resize(11);
//   std::size_t i = 0;
//   for(double x = 0; x <= 2; x+=0.2, ++i) time_mesh[i] = x;
  
//   // define spatial domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3*time_mesh.rows(), 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  
//   // define statistical model
//   double lambdaS = 5.623413*std::pow(0.1, 6); // smoothing in space
//   double lambdaT = 5.623413*std::pow(0.1, 6); // smoothing in time
//   SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatMeshNodes,
// 	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);
//   model.setLambdaS(lambdaS);
//   model.setLambdaT(lambdaT);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile(path_test + "/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.stack(OBSERVATIONS_BLK, y);
//   model.setData(df);
  
//   // // solve smoothing problem
//   model.init();
//   model.solve();

//   // Save C++ solution 
//   DMatrix<double> computedF = model.f();
//   const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   // std::ofstream filef(path_test + "/alpha_" + alpha_string + "/fCpp_" + init_type + ".csv");
//   std::ofstream filef(path_test + "/alpha_" + alpha_string + "/fCpp.csv");
//   if (filef.is_open()){
//         filef << computedF.format(CSVFormatf);
//         filef.close();
//   }

//   // Save n_iter
//   // std::ofstream filen_iter(path_test + "/alpha_" + alpha_string + "/n_iter_" + init_type + ".csv");
//   //   if (filen_iter.is_open()){
//   //     filen_iter << std::setprecision(16) << model.n_iter();
//   //     filen_iter.close();
//   //   }

// }




/* test 2 
   domain:       c-shaped
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
TEST(STQRPDE, Test2_Laplacian_SemiParametric_GeostatisticalAtLocations_Separable_Monolithic) {

  // Parameters 
  const std::string TestNumber = "2"; 

  double alpha = 0.90;
  unsigned int alpha_int = alpha*100; 
  const std::string alpha_string = std::to_string(alpha_int);

  // const std::string init_type = "init_dati"; 
  
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
  double lambdaS = 5.623413*std::pow(0.1, 5); // smoothing in space
  double lambdaT = 5.623413*std::pow(0.1, 5); // smoothing in time
  // load sample position
  CSVReader<double> reader{};
  CSVFile<double> locFile; // locations file
  locFile = reader.parseFile(path_test + "/locs.csv");
  DMatrix<double> loc = locFile.toEigen();

  // Define model 
  SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatLocations,
	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);
  model.setLambdaS(lambdaS);
  model.setLambdaT(lambdaT);
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
  model.solve();

  
  // Save C++ solution 
  DMatrix<double> computedF = model.f();
  const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream filef(path_test + "/alpha_" + alpha_string + "/fCpp_" + init_type + ".csv");
  std::ofstream filef(path_test + "/alpha_" + alpha_string + "/fCpp.csv");
  if (filef.is_open()){
        filef << computedF.format(CSVFormatf);
        filef.close();
  }

  
  // Save C++ solution 
  DMatrix<double> computedFn = model.Psi()*model.f();
  const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream filef(path_test + "/alpha_" + alpha_string + "/fCpp_" + init_type + ".csv");
  std::ofstream filefn(path_test + "/alpha_" + alpha_string + "/fnCpp.csv");
  if (filefn.is_open()){
        filefn << computedFn.format(CSVFormatfn);
        filefn.close();
  }


  // Save beta
  DMatrix<double> computedbeta = model.beta();
  const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  // std::ofstream filef(path_test + "/alpha_" + alpha_string + "/fCpp_" + init_type + ".csv");
  std::ofstream filebeta(path_test + "/alpha_" + alpha_string + "/betaCpp.csv");
  if (filebeta.is_open()){
        filebeta << computedbeta.format(CSVFormatbeta);
        filebeta.close();
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
   time penalization: parabolic (monolithic solution)
 */
TEST(STQRPDE, Test4_Laplacian_SemiParametric_Locations_Parabolic_Monolithic_EstimatedIC) {
  // define time domain, we skip the first time instant because we are going to use the first block of data
  // for the estimation of the initial condition
  DVector<double> time_mesh;
  time_mesh.resize(11);
  for(std::size_t i = 0; i < 10; ++i) time_mesh[i] = 0.4*i;

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square_coarse");
  
  // parabolic PDE
  auto L = dT() + Laplacian();
  
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, time_mesh.rows()); 
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  // define statistical model
  double lambdaS = std::pow(0.1, 4); // smoothing in space
  double lambdaT = std::pow(0.1, 4); // smoothing in time

  double alpha = 0.5; 
  SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeParabolic, fdaPDE::models::GeoStatLocations,
        fdaPDE::models::MonolithicSolver> 
      model(problem, time_mesh, alpha);
  model.setLambdaS(lambdaS);
  model.setLambdaT(lambdaT);


  // load sample position
  CSVReader<double> reader{};
  CSVFile<double> locFile; // locations file
  locFile = reader.parseFile("data/models/STQRPDE/2D_test4/locs.csv");
  DMatrix<double> loc = locFile.toEigen();
  model.set_spatial_locations(loc);

  // load data from .csv files
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile  ("data/models/STQRPDE/2D_test4/y.csv");
  DMatrix<double> y = yFile.toEigen();

  CSVFile<double> XFile; // design matrix
  XFile = reader.parseFile("data/models/STQRPDE/2D_test4/X.csv");
  DMatrix<double> X = XFile.toEigen();

  // set model data
  BlockFrame<double, int> df;
  df.stack(OBSERVATIONS_BLK, y);
  df.stack(DESIGN_MATRIX_BLK, X);
  model.setData(df);

  // define initial condition estimator over grid of lambdas
  InitialConditionEstimator ICestimator(model);
  std::vector<SVector<1>> lambdas_IC;
  for(double x = -9; x <= 3; x += 0.1) lambdas_IC.push_back(SVector<1>(std::pow(10,x))); 
  // compute estimate
  std::cout << "Computing IC..." << std::endl; 
  ICestimator.apply(lambdas_IC);
  DMatrix<double> ICestimate = ICestimator.get();
  std::cout << "End IC computation..." << std::endl; 

  // set estimated initial condition
  model.setInitialCondition(ICestimate);
  model.shift_time(1); // shift data one time instant forward
  
  model.init();
  model.solve();

  const std::string path_test = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/STQRPDE/2D_test4"; 
  // Save C++ solution 
  DMatrix<double> computedF = model.f();
  const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream filef(path_test + "/f.csv");
  if(filef.is_open()){
    filef << computedF.format(CSVFormatf);
    filef.close();
  }

  DMatrix<double> computedFn = model.Psi()*computedF;
  const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream filefn(path_test + "/fn.csv");
  if(filefn.is_open()){
    filefn << computedFn.format(CSVFormatfn);
    filefn.close();
  }

  DMatrix<double> computedBeta = model.beta();
  const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream filebeta(path_test + "/beta.csv");
  if(filebeta.is_open()){
    filebeta << computedBeta.format(CSVFormatbeta);
    filebeta.close();
  }

}







