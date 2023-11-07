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
#include "../fdaPDE/models/regression/MSQRPDE.h"
using fdaPDE::models::MSQRPDE;
#include "../fdaPDE/models/regression/STRPDE.h"
using fdaPDE::models::STRPDE;
#include "../fdaPDE/models/regression/SQRPDE.h"
using fdaPDE::models::SQRPDE;

#include "../fdaPDE/models/SamplingDesign.h"
#include "../../fdaPDE/models/regression/Distributions.h"
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


// TEST(GCV_ozone, US_ozone_GCV) {

//   // path  
//   std::string C_path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/case_study"; 
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study"; 

//   // parameters 
//   bool GCV_optimized = true; 
//   bool parametric = false; 

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("US_mesh");  

//   // load PDE coefficients data
//   CSVReader<double> reader{};
//   auto L = Laplacian();

//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   std::vector<double> alphas = {0.5, 0.75, 0.90, 0.95}; 

//   // Read covariates and locations 
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile (R_path + "/data/X.csv");
//   DMatrix<double> X = XFile.toEigen();
//   CSVFile<double> locFile;
//   locFile = reader.parseFile(R_path + "/data/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();


//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -10.0; x <= -2.0; x +=1.0) lambdas.push_back(SVector<1>(std::pow(10,x)));
//   DVector<double> best_lambda;
//   best_lambda.resize(alphas.size());  


//   std::ofstream fileGCV(R_path + "/data/results/gcv_scores.csv");
//   unsigned int ind = 0; 
//   for(auto alpha : alphas){

//     std::cout << "------------------alpha=" << std::to_string(alpha) << "-----------------" << std::endl; 

//     SQRPDE<decltype(problem), fdaPDE::models::SpaceOnly, 
//             fdaPDE::models::GeoStatLocations, fdaPDE::models::MonolithicSolver> model(problem, alpha);
//     model.set_spatial_locations(loc);

//     // load data from .csv files
//     CSVFile<double> yFile; // observation file
//     yFile = reader.parseFile(R_path + "/data/z.csv");
//     DMatrix<double> y = yFile.toEigen();

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     if(parametric)
//         df.insert(DESIGN_MATRIX_BLK, X);
//     model.setData(df);

//     model.init(); // init model

//     // define GCV function and optimize
//     GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//     GridOptimizer<1> opt;

//     ScalarField<1, decltype(GCV)> obj(GCV);
//     opt.optimize(obj, lambdas); // optimize gcv field
//     best_lambda[ind] = opt.optimum()[0];
        
//     std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda[ind] << std::endl; 
//     ind++;

//     // gcv scores
//     for(std::size_t i = 0; i < GCV.values().size(); ++i){
//         fileGCV << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n"; 
//     }

//   }

//   const static Eigen::IOFormat CSVFormatL(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   std::ofstream fileL(R_path + "/data/results/lambdas_opt.csv");
//   if (fileL.is_open()){
//    fileL << best_lambda.format(CSVFormatL);
//    fileL.close();
//   }

//   fileGCV.close(); 

// }


TEST(MSQRPDE, US_ozone_simulation) {

    // path test  
    std::string C_path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/case_study"; 
    std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study"; 

    // parameters 
    bool GCV_optimized = false; 
    bool parametric = true; 

    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("US_mesh");  

    // load PDE coefficients data
    CSVReader<double> reader{};
    auto L = Laplacian();

    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    std::vector<double> alphas = {0.5, 0.75, 0.90, 0.95}; 

    // Read covariates and locations 
    CSVFile<double> XFile; // design matrix
    XFile = reader.parseFile (R_path + "/data/X.csv");
    DMatrix<double> X = XFile.toEigen();
    CSVFile<double> locFile;
    locFile = reader.parseFile(R_path + "/data/locs.csv");
    DMatrix<double> loc = locFile.toEigen();

    MSQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alphas);
    model.set_spatial_locations(loc);

    CSVFile<double> lFile; // lambdas file
    if(GCV_optimized)
        lFile = reader.parseFile(R_path + "/data/results/lambdas_opt.csv");
    else
        lFile = reader.parseFile(R_path + "/data/results/lambdas.csv");

    DMatrix<double> lambdas = lFile.toEigen();   // the vector should be saved with the "R format"
    model.setLambdas_S(lambdas);
    // load data from .csv files
    CSVFile<double> yFile; // observation file
    yFile = reader.parseFile(R_path + "/data/z.csv");
    DMatrix<double> y = yFile.toEigen();

    // set model data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    if(parametric)
        df.insert(DESIGN_MATRIX_BLK, X);
    model.setData(df);

    // solve smoothing problem
    model.init();
    model.solve();

    // Save solution
    if(!parametric){
        DMatrix<double> computedF = model.f();
        const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filef(R_path + "/data/results/f_all_mult_nonparam.csv");
        if (filef.is_open()){
        filef << computedF.format(CSVFormatf);
        filef.close();
        }

        DMatrix<double> computedFn = model.Psi_mult()*model.f();
        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filefn(R_path + "/data/results/fn_all_mult_nonparam.csv");
        if (filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
        }
    }


    if(parametric){
        DMatrix<double> computedF = model.f();
        const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filef(R_path + "/data/results/f_all_mult.csv");
        if (filef.is_open()){
            filef << computedF.format(CSVFormatf);
            filef.close();
        }

        DMatrix<double> computedFn = model.Psi_mult()*model.f();
        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filefn(R_path + "/data/results/fn_all_mult.csv");
        if (filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
        }

        DMatrix<double> computedBeta = model.beta();
        const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream fileBeta(R_path + "/data/results/beta_all_mult.csv");
        if (fileBeta.is_open()){
            fileBeta << computedBeta.format(CSVFormatBeta);
            fileBeta.close();
        }
    }

}