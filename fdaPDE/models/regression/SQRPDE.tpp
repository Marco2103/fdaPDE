// finds a solution to the SQR-PDE smoothing problem
template <typename PDE, typename SamplingDesign>
void SQRPDE<PDE, SamplingDesign>::solve() {
  // execute FPIRLS for minimization of the functional
  // \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2

  std::cout << "Lambda: " << lambdaS() << std::endl ; 

  FPIRLS<decltype(*this)> fpirls(*this, tol_, max_iter_); // FPIRLS engine

  // matrix_abs_res.resize(n_obs(), max_iter_);
  //  matrix_obs.resize(n_obs(), max_iter_);

  fpirls.compute();
  
  // fpirls converged: extract matrix P and solution estimates
  // W_ = fpirls.weights().asDiagonal();
  W_ = fpirls.solver().W();

  // matrix_abs_res.resize(n_obs(), curr_iter_) ; 

  if(hasCovariates()) {
    XtWX_ = X().transpose()*W_*X(); 
    invXtWX_ = XtWX_.partialPivLu();
  }
  
  // A_ =    fpirls.solver().A(); 
  invA_ = fpirls.solver().invA();
  

  if(hasCovariates()) {
    U_ =    fpirls.solver().U(); 
    V_ =    fpirls.solver().V(); 

  }

  f_ = fpirls.solver().f();
  g_ = fpirls.solver().g();   // per completezza (non sappiamo se lo usa)

  mu_init = fpirls.mu_initialized() ; 

  // matrix_pseudo.resize(n_obs() , max_iter_); 
  // matrix_pseudo = fpirls.matrix_pseudo_fpirls() ;
  
  // matrix_weight.resize(n_obs() , max_iter_); 
  // matrix_weight = fpirls.matrix_weight_fpirls() ; 

  // matrix_f.resize(n_obs() , max_iter_); 
  // matrix_f = fpirls.matrix_f_fpirls() ;

  // matrix_beta.resize(q() , max_iter_); 
  // matrix_beta = fpirls.matrix_beta_fpirls() ;  

  if(hasCovariates()) beta_ = fpirls.solver().beta();
  return;
}


// Non-parametric and semi-parametric cases coincide here, since beta^(0) = 0
template <typename PDE, typename SamplingDesign>
DVector<double> 
SQRPDE<PDE, SamplingDesign>::initialize_mu() const {

  fdaPDE::SparseLU<SpMatrix<double>> invR0_temp ; 
  invR0_temp.compute(R0());

  // assemble system matrix 
  SparseBlockMatrix<double,2,2>
    A_temp(PsiTD()*Psi()/n_obs(), 2*lambdaS()*R1().transpose(),
      lambdaS()*R1(),     -lambdaS()*R0()            );
  // cache non-parametric matrix and its factorization for reuse 
  fdaPDE::SparseLU<SpMatrix<double>> invA_temp ;
  invA_temp.compute( A_temp.derived() );

  DVector<double> b_temp ; 
  b_temp.resize(A_temp.rows());
  b_temp.block(n_basis(),0, n_basis(),1) = 0.*u();
  b_temp.block(0,0, n_basis(),1) = PsiTD()*y()/n_obs() ; 
  BLOCK_FRAME_SANITY_CHECKS;
  DVector<double> f = (invA_temp.solve(b_temp)).head(n_basis());
  DVector<double> fn = Psi()*f ; 

  // implementazione diretta

  // fdaPDE::SparseLU<SpMatrix<double>> invR0_temp ; 
  // invR0_temp.compute(R0());

  // SpMatrix<double> A_temp{} ; 
  // // A_temp = PsiTD()*Psi()/n_obs() + 2*lambdaS()*R1().transpose()*invR0_temp.solve(R1()) ; 
  // A_temp = PsiTD()*Psi() + 2*n_obs()*lambdaS()*R1().transpose()*invR0_temp.solve(R1()) ; 
  // fdaPDE::SparseLU<SpMatrix<double>> invA_temp ;
  // // invA_temp.compute( A_temp.derived() );

  // invA_temp.compute( A_temp );
  
  // DVector<double> b_temp{} ; 
  // // b_temp = PsiTD()*y()/n_obs() ; 
  // b_temp = PsiTD()*y() ; 

  // DVector<double> f = (invA_temp.solve(b_temp)) ;
  // DVector<double> fn = Psi()*f ; 

  return fn ; 
}



template <typename PDE, typename SamplingDesign>
std::tuple<DVector<double>&, DVector<double>&>
SQRPDE<PDE, SamplingDesign>::compute(const DVector<double>& mu) {
  // compute weight matrix and pseudo-observation vector
  DVector<double> abs_res{} ;
  abs_res.resize(y().size()) ; 

//  matrix_abs_res.resize(n_obs(), max_iter_);
//  matrix_obs.resize(n_obs(), max_iter_);

  double tol = 1e-6;    // modificata  

  for(int i = 0; i < y().size(); ++i)
    abs_res(i) = std::abs(y()(i) - mu(i)) ; 
    // abs_res(i) = (std::abs(y()(i) - mu(i)) > tol) ? (std::abs( y()(i) - mu(i) )) : ( std::abs(y()(i) - mu(i)) + tol );   

  // pW_ = (2*n_obs()*abs_res).cwiseInverse() ;  // .matrix();  // aggiunto .matrix()

  // std::cout << "curr_iter in compute is: " << curr_iter_ << std::endl ; 

  // matrix_abs_res.col(curr_iter_) = abs_res; 
  // matrix_obs.col(curr_iter_) = y(); 
  // curr_iter_++;

  // pW_ = (abs_res.array()).inverse().matrix(); //*(1/(2*n_obs()));  
  pW_.resize(n_obs());

  for(int i = 0; i < y().size(); ++i) {
    if (abs_res(i) < tol){
      pW_(i) = ( 1./(abs_res(i)+tol) )/(2.*n_obs());

    }    
    else
      pW_(i) = (1./abs_res(i))/(2.*n_obs()); 
  }
 
  py_ = y() - (1 - 2.*alpha_)*abs_res;
  return std::tie(pW_, py_);
}



template <typename PDE, typename SamplingDesign>
double
SQRPDE<PDE, SamplingDesign>::compute_J_unpenalized(const DVector<double>& mu) {
  
  // compute value of functional J given mu: /(2*n) 
    return (pW_.cwiseSqrt().matrix().asDiagonal()*(py_ - mu)).squaredNorm() ;    // serve .matrix() dopo cwiseSqrt() ? 
    // differentemente da GSRPDE, in cui la sequenza dei calcoli è   
        // array() --> sqrt() --> .. --> matrix() ---> .asDiagonal() 
    // noi chiamiamo cwiseSqrt che opera component wise sul vettore (quindi è lo stesso) e poi chiamiamo la view asDiagonal   
}

// required to support GCV based smoothing parameter selection
// in case of an SRPDE model we have T = \Psi^T*Q*\Psi + \lambda*(R1^T*R0^{-1}*R1)
template <typename PDE, typename SamplingDesign>
const DMatrix<double>& SQRPDE<PDE, SamplingDesign>::T() {
  // compute value of R = R1^T*R0^{-1}*R1, cache for possible reuse
  if(R_.size() == 0){
    invR0_.compute(R0());
    R_ = R1().transpose()*invR0_.solve(R1());
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
const DMatrix<double>& SQRPDE<PDE, SamplingDesign>::Q() {
  // if(Q_.size() == 0){ // Q is computed on request since not needed in general
    // compute Q = W(I - H) = W ( I - X*(X^T*W*X)^{-1}*X^T*W ) 
    Q_ = W()*(DMatrix<double>::Identity(n_obs(), n_obs()) - X()*invXtWX().solve(X().transpose()*W()));
    // W_inQ_ = W() ; 
  // }
  return Q_;
}


// returns the numerator of the GCV score 
template <typename PDE, typename SamplingDesign>
double SQRPDE<PDE, SamplingDesign>::norm
(const DMatrix<double>& fitted, const DMatrix<double>& obs) const {   // CONTROLLA ORDINE degli input 
  double result = 0;
  for(std::size_t i = 0; i < obs.rows(); ++i)
    result += rho_alpha(obs.coeff(i,0) - fitted.coeff(i,0));
  return result*result / n_obs() ;
}

// returns the pinball loss at a specific x 
template <typename PDE, typename SamplingDesign>
double SQRPDE<PDE, SamplingDesign>::rho_alpha(const double& x) const{
  return 0.5*std::abs(x) + (alpha_ - 0.5)*x; 
}


