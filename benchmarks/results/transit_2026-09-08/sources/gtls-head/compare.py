import numpy as np
temp_ootr = np.load('ootr_temp.npy')[:-1]
ootr = np.load('ootr.npy')
# temp =[x + ootr[0] for x in temp_ootr]
temp = []
for i in range(1, len(ootr)):
    if ootr[i] == 0 and temp_ootr[i - 1] == 0:
        temp.append(0)
    else:
        temp.append(ootr[0] + temp_ootr[i -1])

        # temp = ootr[0] + temp_ootr[0]s

temp = [ootr[0]] + temp
print(np.array(temp)[:10])
print(temp_ootr)

print(ootr)
print(np.allclose(temp[:10], ootr[:10]))
print(np.allclose(temp, ootr))