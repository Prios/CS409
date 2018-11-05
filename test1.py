import time

start = time.time()
for i in range(1000):
    print(i)
    time.sleep(0.01)

print('execution time: %.2f' % (time.time() - start))
