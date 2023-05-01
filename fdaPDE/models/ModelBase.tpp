// perform initialization of model object, must be called before call to .solve()
template <typename Model>
void ModelBase<Model>::init(){

  std::cout << "Inizio init" << std::endl ; 

  init_pde();                    // init pde object

  std::cout << "pde fatto" << std::endl ; 

  model().init_regularization(); // init regularization term

  std::cout << "init_reg fatto" << std::endl ; 
  model().init_sampling(true);   // init \Psi matrix, always force recomputation  --> PROBLEMA QUI 
  
std::cout << "init_sam fatto" << std::endl ; 

  // analyze and set missing data
  model().analyze_nan();
  std::cout << "analyz nan fatto" << std::endl ; 

  model().set_nan();
  std::cout << "set nan fatto" << std::endl ; 
  std::cout << "prova init_model " << std::endl ; 
  
  model().init_model();          // init model
}

// a trait to detect if a model requires a preprocessing step
template <typename Model, typename T = void>
struct requires_update_to_data : std::false_type {};

template <typename Model> 
struct requires_update_to_data<
  Model, std::void_t<decltype(std::declval<Model>().update_to_data())>
  > : std::true_type {};

// set model's data from blockframe
template <typename Model>
void ModelBase<Model>::setData(const BlockFrame<double, int>& df) {  
  df_ = df;
  // insert an index row (if not yet present)
  if(!df_.hasBlock(INDEXES_BLK)){
    std::size_t n = df_.rows();
    DMatrix<int> idx(n,1);
    for(std::size_t i = 0; i < n; ++i) idx(i,0) = i;
    df_.insert(INDEXES_BLK, idx);
  }
  // update model to data, if requested
  if constexpr(requires_update_to_data<Model>::value) model().update_to_data();
  return;
}

// analyze missing data pattern, compute nan indices based on observation vector
template <typename Model>
void ModelBase<Model>::analyze_nan() {
  BLOCK_FRAME_SANITY_CHECKS;
  nan_idxs_.clear(); // empty nan vector
  for(std::size_t i = 0; i < n_obs(); ++i){
    if(std::isnan(y()(i,0))){ // requires -ffast-math compiler flag to be disabled, NaN not detected otherwise
      nan_idxs_.emplace_back(i);
      df_.get<double>(OBSERVATIONS_BLK)(i,0) = 0.0; // zero out NaN
    }
  }
  return;
}

// set boundary conditions on problem's linear system
template <typename Model>
void ModelBase<Model>::setDirichletBC(SpMatrix<double>& A, DMatrix<double>& b){
  std::size_t n = A.rows()/2;

  for(std::size_t i = 0; i < n; ++i){
    if(pde_->domain().isOnBoundary(i)){
      A.row(i) *= 0;       // zero all entries of this row
      A.coeffRef(i,i) = 1; // set diagonal element to 1 to impose equation u_j = b_j

      A.row(i+n) *= 0;
      A.coeffRef(i+n,i+n) = 1;

      // boundaryDatum is a pair (nodeID, boundary value)
      double boundaryDatum = pde_->boundaryData().empty() ? 0 : pde_->boundaryData().at(i)[0];
      b.coeffRef(i,0) = boundaryDatum; // impose boundary value
      b.coeffRef(i+n, 0) = 0;
    }
  }
  return;
}
