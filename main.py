import random
import multiprocessing
import io
import numpy as np
import pyopencl as cl
from timing import Timing   #import Timing to calculate time
from algorithm import enum_sort,merge_sort,quick_sort,para_enum_sort,para_merge_sort,para_quick_sort

def read_data(file_name:str)->np.ndarray:
    '''input file_name,return a ndarray of data in file'''
    with open(file_name,"r") as f:
        tmp = [i.rstrip().split(" ") for i in f.readlines()]
        ans = np.array([eval(i) for item in tmp for i in item])
        # print(len(ans))
    return ans

def save_data(file_name:str,data:np.ndarray):
    '''input file_name and data, store data in file with " " split'''
    with open(file_name,"w") as f:
        str_data = [str(i) for i in data] 
        tmp_str=" ".join(str_data)
        f.write(tmp_str)

def test(func,test_num:int):
    '''test im i correct'''
    a = [random.randint(1,20) for i in range(test_num)]
    print(func(np.array(a)))

def diff_test(file1:str,file2):
    '''use diff_test to check whether i am right'''
    with open(file1,'r') as f:
        a = [eval(i) for i in f.readline().rstrip().split(" ")]

    with open(file2,'r') as f:
        b = [eval(i) for i in f.readline().rstrip().split(" ")]
    
    print("check: len({}): {}, len({}): {}".format(file1,len(a),file2,len(b)))
    for i in range(len(a)-1):
        if a[i] != b[i]:
            print("wrong diff!  file1:{}, file2:{}, index:{}".format(file1,file2,i))
    

def check_txt(file:str):
    with open(file,'r') as f:
        a = [eval(i) for i in f.readline().rstrip().split(" ")]
    
    # print("check:the length of data in file is: ",len(a))
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

    #完成基本任务：排序30000个数字
    # timing = Timing()
    # for i in range(2):
        # run_sort("random.txt","order1.txt",quick_sort)
        # run_sort("random.txt","order2.txt",merge_sort)
        # run_sort("random.txt","order3.txt",enum_sort)
        # run_sort("random.txt","order4.txt",para_merge_sort)
        # run_sort("random.txt","order5.txt",para_enum_sort)
        # run_sort("random.txt","order6.txt",para_quick_sort)
    # print(timing)

    #do diff_test
    # for i in range(1,7):
        # diff_test("order1.txt","order"+str(i)+".txt")

    # 测试不同数据集大小下不同实现性能变化
    # timing = Timing()
    # size = 300000
    # name=generate_num("tmp_data/random_50000_{}.txt".format(str(size)),50000,size)
    # run_sort(name,"orderx.txt",quick_sort)
    # run_sort(name,"orderx.txt",merge_sort)
    # run_sort(name,"orderx.txt",para_merge_sort)
    # run_sort(name,"orderx.txt",para_enum_sort)
    # print(timing)
    
    #测试不同重复次数下串行算法和并行算法的差别
    for i in range(4):
        timing = Timing()
        for j in range(2*i+1):
            run_sort("random.txt","orderx.txt",quick_sort)
            run_sort("random.txt","orderx.txt",merge_sort)
            run_sort("random.txt","orderx.txt",para_merge_sort)
            run_sort("random.txt","orderx.txt",para_enum_sort)
        print(timing)
