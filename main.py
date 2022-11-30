import random
import multiprocessing
import io
import numpy as np
import pyopencl as cl
from timing import Timing   #import Timing to calculate time
from algorithm import enum_sort,merge_sort,quick_sort,para_enum_sort,para_merge_sort,para_quick_sort


def test(func,test_num:int):
    '''test im i correct'''
    a = [random.randint(1,20) for i in range(test_num)]
    print(func(np.array(a)))

def check_txt(file:str):
    with open(file,'r') as f:
        a = [eval(i) for i in f.readline().rstrip().split(" ")]
    
    print("check:the length of data in file is: ",len(a))
    for i in range(len(a)-1):
        if a[i] > a[i+1]:
            print("wrong sequnce:{}, {}, index:{}".format(a[i],a[i+1],i))
    
def run_sort(in_file:str,out_file:str,func):
    '''give input file name, output file name, function to run'''
    data = read_data(in_file)
    func = timing(func)
    ans_data = func(data)
    save_data(out_file,ans_data)
    check_txt(out_file)

def generate_num(file:str, num_range:int, num:int):
    '''generate a new test data and store'''
    ans_array = []
    for i in range(num):
        ans_array.append(random.randint(-1*num_range,num_range))
    save_data(file,np.array(ans_array))
    return file


if __name__ == "__main__":

    timing = Timing()
    # name=generate_num("random_50000_300000.txt",50000,300000)
    # run_sort(name,"orderx.txt",quick_sort)
    # run_sort(name,"orderx.txt",merge_sort)
    # run_sort(name,"orderx.txt",para_merge_sort)
    # run_sort(name,"orderx.txt",para_enum_sort)
    # print(timing)
    
    # run_sort("random.txt","order1.txt",quick_sort)
    # run_sort("random.txt","order2.txt",merge_sort)
    # run_sort("random.txt","order3.txt",enum_sort)
    run_sort("random.txt","order4.txt",para_merge_sort)
    # run_sort("random.txt","order5.txt",para_enum_sort)
    # run_sort("random.txt","order6.txt",para_quick_sort)
    print(timing)

    # n = random.randint(5, 10)
    # data = []
    # for i in range(n):
    #     data.append(random.randint(1, 99))
    # ans = para_quick_sort(np.array(data))
    # print(ans)