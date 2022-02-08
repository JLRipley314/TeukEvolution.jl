using TeukEvolution
using Test

include("TestRadial.jl")
include("TestSphere.jl")

import .TestRadial, .TestSphere

#============================================================================
## Testing Radial functions
============================================================================#

# Test 4th order convergence of derivatives

n11, n12 = TestRadial.norms_diff(128,48)
n21, n22 = TestRadial.norms_diff(256,60)

@test n11/n21 >= 3.5 && n11/n21 <= 4.5
@test n12/n22 >= 3.5 && n12/n22 <= 4.5

# Test filter is TVD

n1, n2 = TestRadial.norms_filter(128,48)
n3, n4 = TestRadial.norms_filter(180,90)

@test n2 < n1 
@test n4 < n3 
#============================================================================
## Testing Spherical functions
============================================================================#
#=
TestSphere.test_swal_inner_product(48,-2,-3)
TestSphere.test_swal_inner_product(48, 0,-3)
TestSphere.test_swal_inner_product(48, 1,-3)

TestSphere.test_swal_inner_product(48,-1,-2)
TestSphere.test_swal_inner_product(48, 0,-2)
TestSphere.test_swal_inner_product(48, 2,-2)

TestSphere.test_swal_inner_product(48,-2,-1)
TestSphere.test_swal_inner_product(48, 0,-1)
TestSphere.test_swal_inner_product(48, 1,-1)

TestSphere.test_swal_inner_product(48,-1,0)
TestSphere.test_swal_inner_product(48, 0,0)
TestSphere.test_swal_inner_product(48, 2,0)

TestSphere.test_swal_inner_product(48,-2,1)
TestSphere.test_swal_inner_product(48, 0,1)
TestSphere.test_swal_inner_product(48, 1,1)

TestSphere.test_swal_inner_product(48,-1,2)
TestSphere.test_swal_inner_product(48, 0,2)
TestSphere.test_swal_inner_product(48, 2,2)

TestSphere.test_swal_inner_product(48,-2,3)
TestSphere.test_swal_inner_product(48, 0,3)
TestSphere.test_swal_inner_product(48, 1,3)

TestSphere.test_norm_swal_lap(32, 0,0,2)

TestSphere.test_norm_swal_lap(32, 0,1,2)
TestSphere.test_norm_swal_lap(32, 1,1,2)

TestSphere.test_norm_swal_lap(32,-1,-2,2)
TestSphere.test_norm_swal_lap(32, 0,-2,2)
TestSphere.test_norm_swal_lap(32, 2,-2,2)

TestSphere.test_norm_swal_lap(32,-1,2,2)
TestSphere.test_norm_swal_lap(32, 0,2,2)
TestSphere.test_norm_swal_lap(32, 2,2,2)

TestSphere.test_norm_swal_lap(48,-1,5,23)
TestSphere.test_norm_swal_lap(48, 0,5,23)
TestSphere.test_norm_swal_lap(48, 2,5,23)
=#
TestSphere.test_norm_filter(32,-2,2,2)
TestSphere.test_norm_filter(32,-1,3,4)
TestSphere.test_norm_filter(32, 0,1,5)
TestSphere.test_norm_filter(32, 1,0,3)
TestSphere.test_norm_filter(32, 2,-1,1)
