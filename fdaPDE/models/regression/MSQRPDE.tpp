// perform proper initialization and update of model. Computes quantites which can be reused
// across many calls to solve() and are **not affected by a change in the data**.
// It is implicitly called by ModelBase::init() as part of the initialization process.
// NB: a change in the smoothing parameter must trigger a re-initialization of the model
template <typename PDE, typename SamplingDesign>
void MSQRPDE<PDE, SamplingDesign>::init_model() {

    // Assemble matrices
    assemble_matrices();  

    // Definition of h SQRPDE models for initialization 
    for(std::size_t j = 0; j < h_; ++j){
        SQRPDE<PDE, fdaPDE::models::SpaceOnly, SamplingDesign, fdaPDE::models::MonolithicSolver> model_j(pde(), alphas_[j]);

        // solver initialization
        model_j.data() = data();
        model_j.setLambda(lambdas_[j]);
        model_j.set_spatial_locations(this->locs());
        model_j.init_pde();
        model_j.init_regularization();
        model_j.init_sampling();
        model_j.init_nan();
        model_j.init_model();
        model_j.solve();

        f_curr_(j) = model_j.f();
        fn_curr_(j) = Psi()*model_j.f();
        g_curr_(j) = model_j.g();
        beta_curr_(j) = model_j.beta();

    }
    
    return;
}

  
// finds a solution to the SR-PDE smoothing problem
template <typename PDE, typename SamplingDesign>
void MSQRPDE<PDE, SamplingDesign>::solve() {

    
    SpMatrix<double> W_bar(h_*n_obs(), h_*n_obs());    
    DVector<double> t{};    

    while(crossing_constraints()){

        // algorithm stops when an enought small difference between two consecutive values of the J is recordered
        double J_old = tolerance_+1; double J_new = 0;
        k_ = 0;

        while(k_ < max_iter_ && std::abs(J_new - J_old) > tolerance_){

            // assemble W, Delta, z     
            DVector<double> w_(h_*n_obs());
            DVector<double> delta_((h_-1)*n_obs());
            DVector<double> z_(h_*n_obs());
            for(int j = 0; j < h_; ++j){

                DVector<double> w_j;
                DVector<double> z_j;
                DVector<double> delta_j; 

                if(!hasCovariates()){
                     w_j = 2*(y() - fn_prev_(j)).cwiseAbs(); 

                    if(j < h_-1) {
                        delta_j = 2*(eps_*DVector<double>::Ones(n_obs()) - D_script_.row(j)*fn_prev_(j)).cwiseAbs().cwiseInverse(); 
                    }
                    
                    z_j = y() + (0.5 - alphas_[j])*DiagMatrix<double>(w_j)*DVector<double>::Ones(n_obs()); 

                }
                else{
                    w_j = 2*(y()-X().transpose()*beta_prev_(j) - fn_prev_(j)).cwiseAbs(); 

                    if(j < h_-1) {
                        delta_j = 2*(eps_*DVector<double>::Ones(n_obs())-X().transpose()*D_.row(j)*beta_prev_(j) - D_script_.row(j)*fn_prev_(j)).cwiseAbs().cwiseInverse(); 
                    }
                    
                    z_j = y() + (0.5 - alphas_[j])*DiagMatrix<double>(w_j)*DVector<double>::Ones(n_obs()); 
                }

                w_ << w_ , w_j;
                delta_ << delta_ , delta_j ; 
                z_ << z_, z_j;                 
            }
            DiagMatrix<double> Delta_(delta_);  // non riusciamo a definirla fuori dal ciclo e poi riempirla qui
            W_bar.diagonal() = w_.cwiseInverse(); 
            W_multiple_ = W_bar + gamma_*D_script_.transpose()*Delta_*D_script_; 

            // assemble t 
            t = D_script_.transpose()*Delta_*D_script_*(eps_ + 0.5)*DVector<double>::Ones(h_*n_basis());  

            // assemble system matrix for nonparameteric part
            A_ = SparseBlockMatrix<double,2,2>
            (-Psi_multiple_.transpose()*W_multiple_*Psi_multiple_,    R1_multiple_.transpose(),
              R1_multiple_,                                           R0_multiple_            );
            
            // cache non-parametric matrix factorization for reuse
            invA_.compute(A_);

            // prepare rhs of linear system
            b_.resize(A_.rows());
            b_.block(h_*n_basis(),0, h_*n_basis(),1) = DVector<double>::Zero(h_*n_basis());

            XtWX_multiple_ = X_multiple_.transpose()*W_multiple_*X_multiple_;
            invXtWX_multiple_ = XtWX_multiple_.partialPivLu();

            DVector<double> sol; // room for problem' solution     
            if(!hasCovariates()){ // nonparametric case       
                // update rhs of SR-PDE linear system
                b_.block(0,0, n_basis(),1) = -Psi_multiple_.transpose()*(W_multiple_*z_ + gamma_*t);
                // solve linear system A_*x = b_
                sol = invA_.solve(b_);
                f_curr_ = sol.head(h_*n_basis());
                } else{ // parametric case
                // update rhs of SR-PDE linear system
                b_.block(0,0, n_basis(),1) = -Psi_multiple_.transpose()*(Q_multiple()*z_ + gamma_*(Ihn_ - H_multiple())*t);  
                
                // definition of matrices U and V  for application of woodbury formula
                U_multiple_ = DMatrix<double>::Zero(2*h_*n_basis(), h_*q());
                U_multiple_.block(0,0, h_*n_basis(), h_*q()) = Psi_multiple_.transpose()*W_multiple_*X_multiple_;
                V_multiple_ = DMatrix<double>::Zero(h_*q(), 2*h_*n_basis());
                V_multiple_.block(0,0, h_*q(), h_*n_basis()) = X_multiple_.transpose()*W_multiple_*Psi_multiple_;
                // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
                sol = SMW<>().solve(invA_, U_multiple_, XtWX_multiple_, V_multiple_, b_); 
                // store result of smoothing 
                f_curr_    = sol.head(h_*n_basis());
                beta_curr_ = invXtWX_multiple_.solve((X_multiple_.transpose()*W_multiple_)*(z_ - Psi_multiple_*f_curr_) + 
                                                         gamma_*t);
                }
                // store PDE misfit
                g_curr_ = sol.tail(h_*n_basis());
     
        }

        gamma_ = gamma_*5;  // change factor

    }





  return;
}


template <typename PDE, typename SamplingDesign>
bool MSQRPDE<PDE, SamplingDesign>::crossing_constraints() {
    return true;
}

template <typename PDE, typename SamplingDesign>
const DMatrix<double>& MSQRPDE<PDE, SamplingDesign>::H_multiple() {
  // compute H = X*(X^T*W*X)^{-1}*X^T*W
  H_multiple_ = X_multiple_*invXtWX_multiple_.solve(X_multiple_.transpose()*W_multiple_);

  return H_multiple_;
}

template <typename PDE, typename SamplingDesign>
const DMatrix<double>& MSQRPDE<PDE, SamplingDesign>::Q_multiple() {
  // compute Q = W(I - H) = W ( I - X*(X^T*W*X)^{-1}*X^T*W ) 
  Q_multiple_ = W_multiple_*(DMatrix<double>::Identity(n_obs()*h_, n_obs()*h_) - H_multiple());

  return Q_multiple_;
}
