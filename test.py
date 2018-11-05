import queue
import threading
import multiprocessing
import subprocess
import time

start = time.time()
q = queue.Queue()
for i in range(1000): #put 30 tasks in the queue
    q.put(i)

def worker():
    while True:
        item = q.get()
        #execute a task: call a shell program and wait until it completes
        #subprocess.call("echo "+str(item), shell=True) 
        time.sleep(0.01)
        q.task_done()

cpus=multiprocessing.cpu_count() #detect number of cores
print("Creating %d threads" % cpus)
for i in range(cpus):
     t = threading.Thread(target=worker)
     t.daemon = True
     t.start()

q.join()

print('execution time: %.2f' % (time.time() - start))
