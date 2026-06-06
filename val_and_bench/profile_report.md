# Simulation Profile Report

Generated on: 2026-06-06 23:42:22
Steps: 10000
Stations: 4

## Component Runtime Breakdown

| Component    | Runtime | Time (s) |
| ------------ | ------- | -------- |
| NumPy Overhead |   34.7% |   0.0230s |
| SGP4 & Prop    |   24.1% |   0.0160s |
| Data & Results |   12.4% |   0.0083s |
| Handoff Logic  |   11.0% |   0.0073s |
| Sim Control    |    7.8% |   0.0052s |
| Rain Process   |    1.8% |   0.0012s |
| Link Budget    |    1.8% |   0.0012s |
| Misc Other     |    6.4% |   0.0042s |

## Top 50 Detailed Stats

```

        1    0.005    0.005    0.066    0.066 /home/satyansh/leo_meo/src/satellite_link_sim.py:388(simulate_all_batched)
    10000    0.007    0.000    0.026    0.000 /home/satyansh/leo_meo/src/satellite_link_sim.py:309(select)
    10000    0.003    0.000    0.018    0.000 <__array_function__ internals>:177(argmax)
        1    0.005    0.005    0.018    0.018 /home/satyansh/leo_meo/src/propogate.py:172(get_geometry_batch)
10094/10075    0.003    0.000    0.017    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
    10000    0.005    0.000    0.012    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/fromnumeric.py:1153(argmax)
    10003    0.003    0.000    0.007    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/fromnumeric.py:51(_wrapfunc)
        1    0.000    0.000    0.006    0.006 /home/satyansh/.local/lib/python3.10/site-packages/sgp4/wrapper.py:8(sgp4_array)
        1    0.006    0.006    0.006    0.006 {method '_sgp4' of 'sgp4.vallado_cpp.Satrec' objects}
        4    0.001    0.000    0.004    0.001 {built-in method builtins.any}
    10000    0.004    0.000    0.004    0.000 /home/satyansh/.local/lib/python3.10/site-packages/sgp4/functions.py:8(jday)
        1    0.004    0.004    0.004    0.004 /home/satyansh/leo_meo/src/satellite_link_sim.py:410(<listcomp>)
    10000    0.003    0.000    0.003    0.000 {method 'argmax' of 'numpy.ndarray' objects}
    30035    0.003    0.000    0.003    0.000 /home/satyansh/leo_meo/src/satellite_link_sim.py:541(<genexpr>)
       11    0.001    0.000    0.001    0.000 {method 'tolist' of 'numpy.ndarray' objects}
    30009    0.001    0.000    0.001    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.001    0.001 /home/satyansh/leo_meo/src/satellite_link_sim.py:220(generate_batch)
        1    0.001    0.001    0.001    0.001 /home/satyansh/leo_meo/src/satellite_link_sim.py:150(_simulate_rain_kernel)
       10    0.001    0.000    0.001    0.000 {built-in method numpy.array}
       34    0.001    0.000    0.001    0.000 {built-in method numpy.asanyarray}
       13    0.000    0.000    0.001    0.000 <__array_function__ internals>:177(mean)
       13    0.000    0.000    0.001    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/fromnumeric.py:3345(mean)
       13    0.000    0.000    0.001    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/_methods.py:164(_mean)
    10000    0.001    0.000    0.001    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/fromnumeric.py:1149(_argmax_dispatcher)
    10008    0.001    0.000    0.001    0.000 {built-in method builtins.getattr}
        4    0.001    0.000    0.001    0.000 {built-in method builtins.sorted}
    10016    0.001    0.000    0.001    0.000 {built-in method builtins.len}
        1    0.000    0.000    0.001    0.001 /home/satyansh/leo_meo/src/propogate.py:116(rotate_teme_to_ecef)
        1    0.001    0.001    0.001    0.001 /home/satyansh/leo_meo/src/satellite_link_sim.py:508(<listcomp>)
       32    0.000    0.000    0.000    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        5    0.000    0.000    0.000    0.000 /home/satyansh/leo_meo/src/satellite_link_sim.py:128(effective_path_length)
        4    0.000    0.000    0.000    0.000 <__array_function__ internals>:177(std)
        4    0.000    0.000    0.000    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/fromnumeric.py:3473(std)
        4    0.000    0.000    0.000    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/_methods.py:267(_std)
        4    0.000    0.000    0.000    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/_methods.py:198(_var)
        1    0.000    0.000    0.000    0.000 /home/satyansh/leo_meo/src/propogate.py:106(get_gmst)
        4    0.000    0.000    0.000    0.000 /home/satyansh/leo_meo/src/satellite_link_sim.py:279(packet_loss_from_snr)
       10    0.000    0.000    0.000    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/fromnumeric.py:69(_wrapreduction)
        1    0.000    0.000    0.000    0.000 {method 'normal' of 'numpy.random.mtrand.RandomState' objects}
        1    0.000    0.000    0.000    0.000 /home/satyansh/leo_meo/src/satellite_link_sim.py:541(<listcomp>)
        4    0.000    0.000    0.000    0.000 <__array_function__ internals>:177(amin)
        4    0.000    0.000    0.000    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/fromnumeric.py:2829(amin)
        3    0.000    0.000    0.000    0.000 <__array_function__ internals>:177(stack)
        1    0.000    0.000    0.000    0.000 /home/satyansh/leo_meo/src/propogate.py:144(get_sat_rec)
        1    0.000    0.000    0.000    0.000 <__array_function__ internals>:177(norm)
        1    0.000    0.000    0.000    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/linalg/linalg.py:2342(norm)
       20    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
        5    0.000    0.000    0.000    0.000 /home/satyansh/leo_meo/src/satellite_link_sim.py:253(scintillation_sigma_db)
        3    0.000    0.000    0.000    0.000 /home/satyansh/.local/lib/python3.10/site-packages/numpy/core/shape_base.py:383(stack)
        2    0.000    0.000    0.000    0.000 <__array_function__ internals>:177(einsum)


```
