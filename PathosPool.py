import pathos.multiprocessing as mp
from math import sqrt

def funcA(mylist):
    print('funcA')
    worker_poolA = mp.Pool()
    jobs = mylist
    results = worker_poolA.map(sqrt, jobs)
    worker_poolA.close()
    worker_poolA.join()
    return results

def funcB():
    worker_poolB = mp.ThreadingPool()
    jobs = [[0, 1, 4, 9, 16, 25],[25, 4, 16, 0,1]]
    finalresults = worker_poolB.map(funcA2, jobs)
    worker_poolB.close()
    worker_poolB.join()
    print (finalresults)


def funcA2(mylist):
    print('funcA')
    worker_poolA = mp.Pool()
    # out, indx, jobs = mylist
    output_list = [None for _ in range(len(mylist))]
    inputs = []
    for j in range(len(mylist)):
        inputs.append([mylist[j], output_list, j])
    results = worker_poolA.map(funcT, inputs)
    worker_poolA.close()
    worker_poolA.join()
    print(output_list)
    return output_list

def funcT(params):
    job, output_list, indx = params
    output_list[indx] = sqrt(job)

if __name__ == '__main__':
    funcB()