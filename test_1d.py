"""
Test in one dimension

1. Calculate Kalman Gain
2. Calculate current estimate
3. Calculate new estimate error
"""
mea = [75, 71, 70, 74]
err_mea = 4
initial_est = [68]
prev_err_est = [2]
kg = []

def gain(e_est,e_mea):
    return e_est/(e_est+e_mea)

for i in range(len(mea)):
 
    # Estimate Kalman Gain
    k_gain = gain(prev_err_est[i], err_mea)
    kg.append(k_gain)
   

    est = initial_est[i] + kg[i]*(mea[i]-initial_est[i])
    initial_est.append(est)
   

    err_est = (1-kg[i])*prev_err_est[i]
    prev_err_est.append(err_est)
    

print(kg)
print(initial_est)
print(prev_err_est)

