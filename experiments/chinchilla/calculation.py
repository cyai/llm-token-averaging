
# # 1.5B params, 30B tokens

# 6ND = 270*10^18

# H100 FP16 Flops: 2*10^15
# No Sparsity = 10^15

# Time = 270*10^18 / (2*10^15) = 135000 seconds = 37.5 hours 
# = 270000 s = 75 hours

# MFU: 37.5 hrs * 3 = 112.5 hrs
# 75*3 = 225hrs

# Cost = 112.5 hrs * $2.89 = $325.13 ~ $330 
# = 225*2.89 = $650.25 ~ $650



parms = 10**9
tokens = 20 * 10**9

nd6 = parms * tokens * 6

# p16_flops = 10**15 # h100
# p16_flops = 1.25*10**15 # 8 A6000
p16_flops = 0.989*10**15 # h200

time = nd6 / p16_flops

time_hrs = time / 3600

mfu = time_hrs * 3

cost = mfu * 2.89
print(f"MFU Hours: {mfu:.2f}")
print(f"Cost: ${cost:.2f}")