#ifndef __FEM_STANDARD_SPACE_TIME_SOLVER_H__
#define __FEM_STANDARD_SPACE_TIME_SOLVER_H__

#include "../Assembler.h"
#include "../../utils/Symbols.h"
#include "../PDE.h"
#include "FEMBaseSolver.h"

struct FEMStandardSpaceTimeSolver : public FEMBaseSolver{
  // constructor
  FEMStandardSpaceTimeSolver() = default;

  // solves the PDE using a FEM discretization in space and a finite difference discretization in time (forward-euler scheme)
  template <unsigned int M, unsigned int N, typename E, typename B, typename I> 
  void solve(const PDE<M, N, E>& pde, const B& basis, const I& integrator, double deltaT);
};

// use forward-euler to discretize the time derivative. Under this approximation we get a discretization matrix for the PDE operator
// equal to K = [M/deltaT + A] (forward Euler scheme)
template <unsigned int M, unsigned int N, typename E, typename B, typename I> 
void FEMStandardSpaceTimeSolver::solve(const PDE<M, N, E>& pde, const B& basis, const I& integrator, double deltaT) {
  this->init(pde, basis, integrator);
  // define eigen system solver, use QR decomposition.
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;

  unsigned int timeSteps = this->forcingVector_.cols(); // number of iterations for the time loop
  
  this->solution_.resize(pde.getDomain().getNumberOfNodes(), timeSteps-1);
  this->solution_.col(0) = pde.getInitialCondition(); // impose initial condition
  
  DVector rhs = (this->R0_/deltaT)*pde.getInitialCondition() + this->forcingVector_.col(0);  
  
  // Observe that K is time invariant only for homogeneous boundary conditions. In general we need to recompute K at each time instant, 
  // anyway we can avoid the recomputation of K at each iteration by just keeping overwriting it at the boundary indexes positions. 
  Eigen::SparseMatrix<double> K = this->R0_/deltaT + this->R1_; // build system matrix

  // prepare system matrix to handle dirichlet boundary conditions
  for(std::size_t j = 0; j < pde.getDomain().getNumberOfNodes(); ++j){
    if(pde.getDomain().isOnBoundary(j)){
      K.row(j) *= 0;         // zero all entries of this row
      K.coeffRef(j,j) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
    }
  }
  
  // execute temporal loop to solve ODE system via forward-euler scheme
  for(std::size_t i = 1; i < timeSteps - 1; ++i){
    // impose boundary conditions
    for(std::size_t j = 0; j < pde.getDomain().getNumberOfNodes(); ++j){
      if(pde.getDomain().isOnBoundary(j)){
	// boundaryDatum is a pair (nodeID, boundary time series)
	rhs[j] = pde.getBoundaryData().at(j)[i];; // impose boundary value
      }
    }
    
    solver.compute(K); // prepare solver
    if(solver.info()!=Eigen::Success){ // stop if something was wrong...
      this->success = false;
      return;
    }
    
    DVector u_i = solver.solve(rhs);                               // solve linear system
    this->solution_.col(i) = u_i;                                  // append time step solution to solution matrix
    rhs = (this->R0_/deltaT)*u_i + this->forcingVector_.col(i+1);  // update rhs for next iteration
  }
  return;
}

#endif // __FEM_STANDARD_SPACE_TIME_SOLVER_H__