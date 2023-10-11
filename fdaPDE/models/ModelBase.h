#ifndef __MODEL_BASE_H__
#define __MODEL_BASE_H__

#include <memory>
#include <type_traits>

#include "../core/utils/Symbols.h"
#include "../core/MESH/Mesh.h"
using fdaPDE::core::MESH::Mesh;
#include "../core/MESH/engines/AlternatingDigitalTree/ADT.h"
using fdaPDE::core::MESH::ADT;
#include "../core/utils/DataStructures/BlockFrame.h"
#include "ModelTraits.h"
using fdaPDE::models::model_traits;
#include "ModelMacros.h"

namespace fdaPDE {
namespace models {
   
  // abstract base interface for any fdaPDE statistical model. Uses CRTP pattern
  template <typename Model>
  class ModelBase {
  public:
    typedef typename model_traits<Model>::PDE PDE; // PDE used for regularization in space
    static constexpr std::size_t M = PDE::local_dimension;
    static constexpr std::size_t N = PDE::embedding_dimension;
    static constexpr std::size_t K = PDE::basis_order;
    
    // constructor
    ModelBase() = default;
    ModelBase(const PDE& pde) : pde_(std::make_shared<PDE>(pde)) {};
    // copy constructor
    ModelBase(const ModelBase& rhs) { pde_ = rhs.pde_; }
    void init_pde() { pde_->init(); };
    void init(); // entry point for full model initialization
    
    // setters
    void setDirichletBC(SpMatrix<double>& A, DMatrix<double>& b);
    void setData(const BlockFrame<double, int>& df, bool reindex = false); // initialize model's data by copying the supplied BlockFrame
    BlockFrame<double, int>& data() { return df_; }  // direct write-access to model's internal data storage
    void setLambda(const SVector<model_traits<Model>::n_lambda>& lambda) { 
      
      lambda_ = lambda; 
      
      }
    void setPDE(const PDE& pde) { pde_ = std::make_shared<PDE>(pde); }
    
    // getters
    const BlockFrame<double, int>& data() const { return df_; }
    const DMatrix<int>& idx() const { return df_.get<int>(INDEXES_BLK); } // data indices
    // informations related to discretization of regularization term
    const PDE& pde() const { return *pde_; } // regularizing term Lf - u (defined on some domain \Omega)
    const Mesh<M,N,K>& domain() const { return pde_->domain(); }
    std::size_t n_basis() const { return pde_->domain().dof(); }; // number of basis functions used in space discretization
    std::size_t n_locs() const { return model().n_spatial_locs()*model().n_temporal_locs(); } // number of observations' locations
    const ADT<M,N,K>& gse() { if(gse_ == nullptr){ gse_ = std::make_shared<ADT<M,N,K>>(domain()); } return *gse_; }
    SVector<model_traits<Model>::n_lambda> lambda() const { return lambda_; }
    
    // abstract part of the interface, must be implemented by concrete models
    virtual void solve() = 0; // finds a solution to the problem, whatever the problem is.
    // destructor
    virtual ~ModelBase() = default;
  protected:   
    std::shared_ptr<PDE> pde_; // regularizing term Lf - u and domain definition D
    std::shared_ptr<ADT<M,N,K>> gse_; // geometric search engine
    BlockFrame<double, int> df_; // blockframe for data storage
    SVector<model_traits<Model>::n_lambda> lambda_; // vector of smoothing parameters
    
    // getter to underlying model object
    inline Model& model() { return static_cast<Model&>(*this); }
    inline const Model& model() const { return static_cast<const Model&>(*this); } // const version
  };

  // macro for runtime sanity checks on data, should be the first instruction in a solve() implementation
#define BLOCK_FRAME_SANITY_CHECKS					\
  /* raise error if data comes without index block */			\
  if(!data().hasBlock(INDEXES_BLK))					\
    throw std::logic_error("bad BlockFrame, no index block found");	\
  /* stop if incoming data has no observations */			\
  if(!data().hasBlock(OBSERVATIONS_BLK))				\
    throw std::logic_error("bad BlockFrame, model without observations" \
			   " is ill-formed");				\
  
  #include "ModelBase.tpp"

}}

#endif // __MODEL_BASE_H__
