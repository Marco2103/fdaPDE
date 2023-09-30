## load all required modules

## regularizing PDES
loadModule("Laplacian_2D_Order1", TRUE)
loadModule("Laplacian_3D_Order1", TRUE)
loadModule("Laplacian_1_5D_Order1", TRUE)
loadModule("ConstantCoefficients_2D_Order1", TRUE)
loadModule("SpaceVarying_2D_Order1", TRUE)
loadModule("Laplacian_2_5D_Order1", TRUE)    # new
loadModule("Laplacian_1_5D_Order1", TRUE)    # new

## SRPDE
# loadModule("SRPDE_Laplacian_2D_GeoStatNodes", TRUE)
# loadModule("SRPDE_Laplacian_2D_GeoStatLocations", TRUE)
# loadModule("SRPDE_Laplacian_2D_Areal", TRUE)
# loadModule("SRPDE_ConstantCoefficients_2D_GeoStatNodes", TRUE)
# loadModule("SRPDE_ConstantCoefficients_2D_GeoStatLocations", TRUE)
# loadModule("SRPDE_ConstantCoefficients_2D_Areal", TRUE)
# loadModule("SRPDE_SpaceVarying_2D_GeoStatNodes", TRUE)
# loadModule("SRPDE_SpaceVarying_2D_GeoStatLocations", TRUE)
# loadModule("SRPDE_SpaceVarying_2D_Areal", TRUE)


## SQRPDE
loadModule("SQRPDE_Laplacian_2D_GeoStatNodes", TRUE)
loadModule("SQRPDE_Laplacian_2D_GeoStatLocations", TRUE)
loadModule("SQRPDE_Laplacian_2D_Areal", TRUE)
loadModule("SQRPDE_ConstantCoefficients_2D_GeoStatNodes", TRUE)
loadModule("SQRPDE_Laplacian_3D_GeoStatLocations", TRUE)
loadModule("SQRPDE_Laplacian_2_5D_GeoStatNodes", TRUE)
loadModule("SQRPDE_Laplacian_1_5D_GeoStatNodes", TRUE)