
def aggregated_contribution(contribution_map):
    contr_sum  = {}
    for j, dct in enumerate(contribution_map):
        for k in set(dct.keys()).union(set(contr_sum.keys())):
            contr_sum[k] = (contr_sum.get(k, 0)*j + dct.get(k,0) ) / (j+1)
    return contr_sum