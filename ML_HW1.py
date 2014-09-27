import numpy as np

def targetf():  #this function obtains a random m and b for a linear function
    x1 = np.random.uniform(-1,1)
    x2 = np.random.uniform(-1,1)
    y1 = np.random.uniform(-1,1)
    y2 = np.random.uniform(-1,1)
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return m, b

def mapping(x,f):   #this function takes a 2D point and compares is to the target function f and returns y=1 or y=-1
    if x[1]<f(x=x[0]):
        return -1
    else:
        return 1

def dataset(N):
    D =[]
    for i in xrange(N):
        x1 = np.random.uniform(-1,1)
        x2 = np.random.uniform(-1,1)
        D.append([x1,x2])
    return np.array(D)

def h(w,x): #hypothesis function
	return (np.sign(sum(w*x)))

def PLA(N, times):

	iterations = np.zeros(times)
    for t in xrange(times):

		m, b = targetf();
		f = lambda x: m*x+b
		D = dataset(N)
		t_set = []
		w = [0,0,0]
		local_iterations = 0
		iterate = True

		for i in D:
	    	t_set.append(mapping(i,f))

		t_set = np.array(t_set)

	    while iterate:
	    	tt = []
	    	for i in D:
	    		tt.append(h(w,np.array([1,i[0],i[1]])))
	    	tt = np.array(tt)

	    	miss = D[t_set!=tt]  # misclassified points
	    	miss = np.array(miss)
	    	if miss.size > 0:
	    		item = np.random.choice(range(len(miss)))
	    		w = w + mapping(miss[item],f)*np.array([1,miss[item][0],miss[item][1]])
	    		local_iterations = local_iterations + 1
			else:
				break

		iterations[t]=local_iterations
	
	return iterations






