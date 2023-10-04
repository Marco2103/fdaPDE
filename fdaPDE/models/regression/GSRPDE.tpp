// finds a solution to the GSR-PDE smoothing problem
template <typename PDE, typename RegularizationType, typename SamplingDesign,
	  typename Solver, typename Distribution>
void GSRPDE<PDE, RegularizationType, SamplingDesign, Solver, Distribution>::solve() {

  // execute FPIRLS for minimization of the functional
  // \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2
  FPIRLS<decltype(*this), Distribution> fpirls(*this, tol_, max_iter_); // FPIRLS engine
  fpirls.compute();
  
  // fpirls converged: extract matrix P and solution estimates
  W_ = fpirls.solver().W();

  if(hasCovariates()) {           
    XtWX_ = X().transpose()*W_*X(); 
    invXtWX_ = XtWX_.partialPivLu();
  }

  f_ = fpirls.solver().f();
  if(hasCovariates()) beta_ = fpirls.solver().beta();
  return;
}

template <typename PDE, typename RegularizationType, typename SamplingDesign,
	  typename Solver, typename Distribution>
std::tuple<DVector<double>&, DVector<double>&>
GSRPDE<PDE, RegularizationType, SamplingDesign, Solver, Distribution>::compute(const DVector<double>& mu){
  DVector<double> theta_ = distribution_.link(mu); // \theta^k = [ g(\mu^k_1), ..., g(\mu^k_n) ]
  DVector<double> G_ = distribution_.der_link(mu); // G^k = diag(g'(\mu^k_1), ..., g'(\mu^k_n))
  DVector<double> V_ = distribution_.variance(mu); // V^k = diag(v(\mu^k_1), ..., v(\mu^k_n))
  // compute weight matrix and pseudo-observation vector
  pW_ = ((G_.array().pow(2)*V_.array()).inverse()).matrix();
  py_ = G_.asDiagonal()*(y() - mu) + theta_;
  return std::tie(pW_, py_);
}


template <typename PDE, typename RegularizationType, typename SamplingDesign,
	  typename Solver, typename Distribution>
DVector<double>
GSRPDE<PDE, RegularizationType, SamplingDesign, Solver, Distribution>::initialize_mu() const {
  return y();
}

template <typename PDE, typename RegularizationType, typename SamplingDesign,
	  typename Solver, typename Distribution>
double
GSRPDE<PDE, RegularizationType, SamplingDesign, Solver, Distribution>::model_loss(const DVector<double>& mu) const{
  
  // compute value of functional J given mu: \norm{V^{-1/2}(y - \mu)}^2 
	DVector<double> V = distribution_.variance(mu).array().sqrt().inverse().matrix();
	double J = (V.asDiagonal()*(y() - mu)).squaredNorm() ;
  return J;
}

// required to support GCV based smoothing parameter selection
// in case of a GSRPDE model we have T = \Psi^T*Q*\Psi + \lambda*(R1^T*R0^{-1}*R1), with Q = W*(I-H)
template <typename PDE, typename RegularizationType, typename SamplingDesign,
	  typename Solver, typename Distribution>
const DMatrix<double>& GSRPDE<PDE, RegularizationType, SamplingDesign, Solver, Distribution>::T() {
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
template <typename PDE, typename RegularizationType, typename SamplingDesign,
	  typename Solver, typename Distribution>
const DMatrix<double>& GSRPDE<PDE, RegularizationType, SamplingDesign, Solver, Distribution>::Q() {
  if(Q_.size() == 0){ // Q is computed on request since not needed in general
    // compute Q = W(I - H) = W - W*X*(X*W*X^T)^{-1}*X^T*W
    Q_ = W()*(DMatrix<double>::Identity(n_obs(), n_obs()) - X()*invXtWX().solve(X().transpose()*W()));
  }
  return Q_;
}

// returns the deviance of y - \hat y induced by the specific distribution considered.
template <typename PDE, typename RegularizationType, typename SamplingDesign,
	  typename Solver, typename Distribution>
double GSRPDE<PDE, RegularizationType, SamplingDesign, Solver, Distribution>::norm
(const DMatrix<double>& fitted, const DMatrix<double>& obs) const {   
  Distribution distr_{}; // define distribution object
  // total deviance computation

  // compute mu to pass to the deviance function
  DMatrix<double> mu = distr_.inv_link(fitted);
  double result = 0;
  for(std::size_t i = 0; i < obs.rows(); ++i) 
    result += distr_.deviance(mu.coeff(i,0), obs.coeff(i,0));
    
  return result;
}
