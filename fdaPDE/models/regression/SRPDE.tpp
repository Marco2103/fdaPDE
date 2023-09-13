// perform proper initialization and update of model. Computes quantites which can be reused
// across many calls to solve() and are **not affected by a change in the data**.
// It is implicitly called by ModelBase::init() as part of the initialization process.
// NB: a change in the smoothing parameter must trigger a re-initialization of the model
template <typename PDE, typename SamplingDesign>
void SRPDE<PDE, SamplingDesign>::init_model() {
  if(LinearSystemType_ == "Woodbury"){
    // assemble system matrix for nonparameteric part
    A_ = SparseBlockMatrix<double,2,2>
      (-PsiTD()*W()*Psi(), lambdaS()*R1().transpose(),
      lambdaS()*R1(),     lambdaS()*R0()            );
    
    // cache non-parametric matrix factorization for reuse
    //std::cout << "LU factorization for invA" << std::endl; 
    invA_.compute(A_);
    // prepare rhs of linear system
    b_.resize(A_.rows());
    b_.block(n_basis(),0, n_basis(),1) = lambdaS()*u();
    return;

  } else return;

}
  
// finds a solution to the SR-PDE smoothing problem
template <typename PDE, typename SamplingDesign>
void SRPDE<PDE, SamplingDesign>::solve() {

  if(LinearSystemType_ == "Woodbury"){   // M: nonparametric: block resolution; semiparametric: SMW
  BLOCK_FRAME_SANITY_CHECKS;
  DVector<double> sol; // room for problem' solution 
  if(!hasCovariates()){ // nonparametric case       
      // update rhs of SR-PDE linear system
      b_.block(0,0, n_basis(),1) = -PsiTD()*W()*y();
      // solve linear system A_*x = b_
      sol = invA_.solve(b_);
      f_ = sol.head(n_basis());
    }else{ // parametric case
      // update rhs of SR-PDE linear system
      b_.block(0,0, n_basis(),1) = -PsiTD()*lmbQ(y()); // -\Psi^T*D*Q*z
      
      // definition of matrices U and V  for application of woodbury formula
      U_ = DMatrix<double>::Zero(2*n_basis(), q());
      U_.block(0,0, n_basis(), q()) = PsiTD()*W()*X();
      V_ = DMatrix<double>::Zero(q(), 2*n_basis());
      V_.block(0,0, q(), n_basis()) = X().transpose()*W()*Psi();
      // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
      sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_); 
      // store result of smoothing 
      f_    = sol.head(n_basis());
      beta_ = invXtWX().solve(X().transpose()*W())*(y() - Psi()*f_);
    }
    // store PDE misfit
    g_ = sol.tail(n_basis());
  } 
  else{  // M: direct resolution with Cholesky
    if(!hasCovariates()){ // nonparametric case  
      SpMatrix<double> A_Cholesky = PsiTD()*W()*Psi() + Base::pen(); 
      CholFactorization invA_Cholesky;
      invA_Cholesky.compute(A_Cholesky); 
      b_Chol_.resize(A_Cholesky.rows());

      if(Base::pde_ -> massLumpingSystem()){
        DVector<double> lumped_vector;
        DiagMatrix<double> lumped_invR0;
        lumped_vector.resize(n_basis()); 
        for(std::size_t j = 0; j < n_basis(); ++j)  
          lumped_vector[j] = 1 / R0().col(j).sum();  
        lumped_invR0 = lumped_vector.asDiagonal();

        b_Chol_ = lambdaS()*R1().transpose()*lumped_invR0*u() + PsiTD()*W()*y(); 
        f_ = invA_Cholesky.solve(b_Chol_); 
        g_ = lumped_invR0*(R1()*f_ - u()); 
      } else{
        fdaPDE::SparseLU<SpMatrix<double>> invR0;
        invR0.compute(Base::pde_->R0());

        b_Chol_ = lambdaS()*R1().transpose()*invR0.solve(u()) + PsiTD()*W()*y(); 
        f_ = invA_Cholesky.solve(b_Chol_);  
        g_ = invR0.solve(R1()*f_- u()); 
      }   
      
    } else{ // parametric case  (same as semiparametric but with Q in place of W (see Sangalli ISR pag. 515))

      SpMatrix<double> A_Cholesky = PsiTD()*Base::lmbQ(Psi()) + Base::pen(); 
      CholFactorization invA_Cholesky;
      invA_Cholesky.compute(A_Cholesky);
      b_Chol_.resize(A_Cholesky.rows());

      if(Base::pde_ -> massLumpingSystem()){
        DVector<double> lumped_vector;
        DiagMatrix<double> lumped_invR0;
        lumped_vector.resize(n_basis()); 
        for(std::size_t j = 0; j < n_basis(); ++j)  
          lumped_vector[j] = 1 / R0().col(j).sum();  
        lumped_invR0 = lumped_vector.asDiagonal();

        b_Chol_ = lambdaS()*R1().transpose()*lumped_invR0*u() + PsiTD()*Base::lmbQ(y()); 
        f_ = invA_Cholesky.solve(b_Chol_);
        beta_ = invXtWX().solve(X().transpose()*W())*(y() - Psi()*f_); 
        g_ = lumped_invR0*(R1()*f_ - u()); 
        
      } else{
        fdaPDE::SparseLU<SpMatrix<double>> invR0;
        invR0.compute(Base::pde_->R0());

        b_Chol_ = lambdaS()*R1().transpose()*invR0.solve(u()) + PsiTD()*Base::lmbQ(y());
        f_ = invA_Cholesky.solve(b_Chol_); 
        beta_ = invXtWX().solve(X().transpose()*W())*(y() - Psi()*f_);
        g_ = invR0.solve(R1()*f_ - u());
      }



 
    }
  }

  return;
}

// required to support GCV based smoothing parameter selection
// in case of an SRPDE model we have T = \Psi^T*Q*\Psi + \lambda*(R1^T*R0^{-1}*R1)
template <typename PDE, typename SamplingDesign>
const DMatrix<double>& SRPDE<PDE, SamplingDesign>::T() {
  // compute value of R = R1^T*R0^{-1}*R1, cache for possible reuse
  if(R_.size() == 0){
    if(!massLumpingGCV()){
      invR0_.compute(R0());
      R_ = R1().transpose()*invR0_.solve(R1());
    } else{
        DVector<double> lumped_invR0;
        lumped_invR0.resize(n_basis()); 
        for(std::size_t j = 0; j < n_basis(); ++j)  
          lumped_invR0[j] = 1 / R0().col(j).sum();  
        lumped_invR0_ = lumped_invR0.asDiagonal();
        R_ = R1().transpose()*lumped_invR0_*R1();
    }
  }
  // compute and store matrix T for possible reuse
  if(!hasCovariates()) // case without covariates, Q is the identity matrix
    T_ = PsiTD()*W()*Psi()   + lambdaS()*R_;
  else // general case with covariates
    T_ = PsiTD()*lmbQ(Psi()) + lambdaS()*R_;
  return T_;
}

// Q is computed on demand only when it is needed by GCV and cached for fast reacess (in general operations
// involving Q can be substituted with the more efficient routine lmbQ(), which is part of iRegressionModel interface)
template <typename PDE, typename SamplingDesign>
const DMatrix<double>& SRPDE<PDE, SamplingDesign>::Q() {
  // if(Q_.size() == 0){ // Q is computed on request since not needed in general
    // compute Q = W(I - H) = W - W*X*(X*W*X^T)^{-1}*X^T*W
  Q_ = W()*(DMatrix<double>::Identity(n_obs(), n_obs()) - X()*invXtWX().solve(X().transpose()*W()));
  // }
  return Q_;
}

// returns the euclidean norm of op1 - op2
template <typename PDE, typename SamplingDesign>
double SRPDE<PDE, SamplingDesign>::norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const {
  return (op1 - op2).squaredNorm();
}
