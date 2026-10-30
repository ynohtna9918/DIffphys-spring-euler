from src.mass_spring import SpringSystem


"""
Test: nomalization and validation
"""
base = dict(x=[[0,0,0],[1.1,0,0]], v=[[0,0,0],[0,0,0]], inv_masses=[0,1],
            springs=[[0,1]], rest_lengths=[1.0], k_s=100, k_d=0,
            gravity=[0,0,0], drag=0)
def bad(**change):
    try:
        SpringSystem(**{**base, **change}); print("NO ERROR (check missing!)")
    except ValueError as e:
        print("ValueError:", e)

bad(x=[0, 0, 0])                         # 1: x not 2-D
bad(v=[[0,0,0]])                         # 2: v shape (1,3) vs (2,3)
bad(inv_masses=[0, 1, 1])                # 3: inv_masses (3,) vs N=2
bad(inv_masses=[0, -1])                  # 4: negative inverse mass
bad(springs=[0, 1])                      # 5: springs not (M, 2)
bad(springs=[[0, 2]])                    # 6: index 2 out of range for N=2
bad(springs=[[1, 1]])                    # 7: self-spring
bad(rest_lengths=[1.0, 1.0])             # 8: rest_lengths (2,) vs M=1
bad(rest_lengths=[0.0])                  # 9: rest length not > 0
bad(gravity=[0, -9.81])                  # 10: gravity (2,) vs dim=3
bad(k_s=-5)                              # 11: negative stiffness
bad(x=[[0,0,0],[float("nan"),0,0]])      # 12: NaN in x