"""Minimum follower displacement with selected actuator priority (convex projection)."""
from itertools import combinations
import numpy as np

def coordinated_move(limits, exposure_m, selected, amount_mm):
    current=limits.validate(exposure_m)*1000
    target=float(np.clip(current[selected]+amount_mm,
                         limits.lower_m[selected]*1000,limits.upper_m[selected]*1000))
    # A x <= b in material exposure coordinates; equivalently measured carriage bounds/gaps.
    A=np.vstack((np.eye(3),-np.eye(3),[1,-1,0],[-1,1,0],[0,1,-1],[0,-1,1]))
    b=np.r_[np.asarray(limits.upper_m)*1000,-np.asarray(limits.lower_m)*1000,
            100,-getattr(limits,'inner_difference_min',0)*1000,10,0]
    if hasattr(limits,'linear_constraints_mm'):A,b=limits.linear_constraints_mm
    others=[i for i in range(3) if i!=selected]
    B=A[:,others]; c=b-A[:,selected]*target; old=current[others]
    candidates=[old]
    for row,rhs in zip(B,c):
        norm=row@row
        if norm>0:candidates.append(old-row*((row@old-rhs)/norm))
    for j,k in combinations(range(len(c)),2):
        matrix=B[[j,k]]
        if abs(np.linalg.det(matrix))>1e-10:
            candidates.append(np.linalg.solve(matrix,c[[j,k]]))
    feasible=[x for x in candidates if np.all(B@x<=c+1e-8)]
    if not feasible:raise ValueError('No feasible coordinated movement')
    result=current.copy(); result[selected]=target
    result[others]=min(feasible,key=lambda x:np.sum((x-old)**2))
    result[np.abs(result)<1e-9]=0
    return limits.validate(result/1000)
