import random
import multiprocessing
import io
import numpy as np
import pyopencl as cl
from timing import Timing   #import Timing to calculate time
# Personal implementation of serial partitioning and quicksort


timing = Timing()

def read_data(file_name:str)->np.ndarray:
    '''input file_name,return a ndarray of data in file'''
    with open(file_name,"r") as f:
        tmp = [i.rstrip().split(" ") for i in f.readlines()]
        ans = np.array([eval(i) for item in tmp for i in item])
        print(len(ans))
    return ans

def save_data(file_name:str,data:np.ndarray):
    '''input file_name and data, store data in file with " " split'''
    with open(file_name,"w") as f:
        str_data = [str(i) for i in data] 
        tmp_str=" ".join(str_data)
        f.write(tmp_str)

def merge(data:np.ndarray)->np.ndarray:
    '''serially merge sort data'''
    if len(data) <= 1:
        return data
    mid = len(data)//2 # mid>=1
    left = merge(data[0:mid])
    right = merge(data[mid:])

    l_point = 0
    r_point = 0
    tol = len(data)
    ans = []
    while l_point+r_point < tol:
        if l_point >= mid:
            ans.append(right[r_point])
            r_point += 1
        
        elif r_point >= len(data)-mid:
            ans.append(left[l_point])
            l_point += 1
        
        else:
            if left[l_point] < right[r_point]:
                ans.append(left[l_point])
                l_point += 1
            else:
                ans.append(right[r_point])
                r_point += 1
    return np.array(ans) 

@timing
def merge_sort(data:np.ndarray)->np.ndarray:
    return merge(data)

def quick(data:np.ndarray, left:int, right:int)->np.ndarray:
    '''sort data[left:right],inplace'''
    if  right <= left+1:
        return
    pivot = data[random.randint(left,right-1)]  # get a flag by random sample index of data
    equal = left
    bigger = right
    s_num = left
    while s_num < bigger:
        tmp = data[s_num]
        if tmp < pivot:
            data[s_num] = data[equal]
            data[equal] = tmp
            equal += 1
            s_num += 1
        elif tmp == pivot:
            s_num += 1
        else:
            data[s_num] = data[bigger-1]
            data[bigger-1] = tmp
            bigger -= 1
    quick(data,left,equal)
    quick(data,bigger,right)
    return data

@timing
def quick_sort(data:np.ndarray)->np.ndarray:
    '''serially quick sort data'''
    quick(data,0,len(data))
    return data

@timing
def enum_sort(data:np.ndarray)->np.ndarray:
    '''serial enum sort,注这里似乎用了浮点数，希望不影响'''
    #注：对于相同的元素，定义index小的将排在前面（这样可以完全有序）
    ans = np.zeros(len(data))
    for i in range(len(data)):
        tmp_data = data[i]
        index = 0
        for j in range(len(data)):
            if data[j] < tmp_data:
                index +=1
            elif data[j] == tmp_data and j < i:
                index +=1
        ans[index] = tmp_data
    return ans

@timing
def para_merge_sort(data_in:np.ndarray)->np.ndarray:
    '''parallel compute merge sort'''
    # OpenCL kernel 函數程式碼
    CL_CODE = '''
    kernel void merge(int chunk_size, int size, global long* data, global long* buff) {
        // 取得分組編號
        const int gid = get_global_id(0);

        // 根據分組編號計算責任範圍
        const int offset = gid * chunk_size;
        const int real_size = min(offset + chunk_size, size) - offset;
        global long* data_part = data + offset;
        global long* buff_part = buff + offset;

        // 設定合併前的初始狀態
        int r_beg = chunk_size >> 1;
        int b_ptr = 0;
        int l_ptr = 0;
        int r_ptr = r_beg;

        // 進行合併
        while (b_ptr < real_size) {
            if (r_ptr >= real_size) {
                // 若右側沒有資料，取左側資料堆入緩衝區
                buff_part[b_ptr] = data_part[l_ptr++];
            } else if (l_ptr == r_beg) {
                // 若左側沒有資料，取右側資料堆入緩衝區
                buff_part[b_ptr] = data_part[r_ptr++];
            } else {
                // 若兩側都有資料，取較小資料堆入緩衝區
                if (data_part[l_ptr] < data_part[r_ptr]) {
                    buff_part[b_ptr] = data_part[l_ptr++];
                } else {
                    buff_part[b_ptr] = data_part[r_ptr++];
                }
            }
            b_ptr++;
        }
    }
    '''


    # create context, queue, program
    plat = cl.get_platforms()
    devices = plat[0].get_devices()
    ctx = cl.Context([devices[0]])
    prg = cl.Program(ctx, CL_CODE).build()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # 資料轉換成 numpy 形式以利轉換為 OpenCL Buffer
    data_np = np.int64(data_in)
    buff_np = np.empty_like(data_np).astype(np.int64)

    # 建立緩衝區，並且複製數值到緩衝區
    data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data_np)
    buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=buff_np)

    # 設定合併前初始狀態
    data_len = np.int32(len(data_np))
    chunk_size = np.int32(1)

    while chunk_size < data_len:
        # 更新分組大小，每一回合變兩倍
        chunk_size <<= 1
        # 換算平行作業組數 
        group_size = ((data_len - 1) // chunk_size) + 1
        # 進行分組合併作業
        prg.merge(queue, (group_size,), (1,), chunk_size, data_len, data, buff)
        # 將合併結果作為下一回合的原始資料
        temp = data
        data = buff
        buff = temp
        # 顯示此回合狀態
        cl.enqueue_copy(queue, data_np, data)

    queue.finish()
    data.release()
    buff.release()
    return data_np

def bad_para_quick_sort(data_in:np.ndarray)->np.ndarray:
    '''need synchronize? hard!'''
    CL_CODE = '''
        __kernel void sort_tree(global long* data, global int* f, 
        global int* l_child, global int* r_child, global bool* existed) {
            //get id
            const int gid = get_global_id(0)

            if(existed[gid]) return;

            bool is_left = false;
            if(data[gid]<data[f[gid]] || (data[gid]==data[f[gid]] && gid<f[gid])) {
                l_child[f[gid]] = gid;
                is_left = true;
            }
            else {
                r_child[f[gid]] = gid;
                is_left = false;
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            if(is_left) {
                if(l_child[f[gid]] == gid) {
                    existed[gid] = true;
                }
                else {
                    f[gid] = l_child[f[gid]];
                }
            }
            else {
                if(r_child[f[gid]] == gid) {
                    existed[gid] = true;
                }
                else {
                    f[gid] = r_child[f[gid]];
                }
            }
        } 
    '''

    # create context, queue, program
    plat = cl.get_platforms()
    devices = plat[0].get_devices()
    ctx = cl.Context([devices[0]])
    prg = cl.Program(ctx, CL_CODE).build()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # change data to np.int64, create OpenCL Buffer
    data_np = np.int64(data_in)
    father_np = np.empty_like(data_np,dtype=np.int32)
    l_child_np = np.full_like(data_np,len(data_in),dtype=np.int32)
    r_child_np = np.full_like(data_np,len(data_in),dtype=np.int32)
    existed_np = np.empty_like(data_np,dtype=np.bool_)

    #create buffer in device
    data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data_np)
    father = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=father_np)
    l_child = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=l_child_np)
    r_child = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=r_child_np)
    existed = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=existed_np)

    # figure originate state
    data_len = np.int32(len(data_np))

    global_size = ((len(data_np)),) #not sure but seems to be all
    prg.enumerate(queue, global_size, global_size, data_len, data, father,l_child,r_child,existed)

    #copy buff from device to host
    cl.enqueue_copy(queue, data_np, buff)

    queue.finish()
    data.release()
    buff.release()
    return data_np

def partition(data:np.ndarray,pivot):
    '''a partition of original list'''
    # Assert that no parameter can be "None"
    assert data is not None
    assert pivot is not None

    l_data = np.ndarray([i for i in data if i<pivot])
    m_data = np.ndarray([i for i in data if i== pivot])
    r_data = np.ndarray([i for i in data if i> pivot])

    return l_data,m_data,r_data


def parallel_quicksort(data:np.ndarray, n_socket, proc_count, MAX_PROCESSES_COUNT):
    '''parallel quicksort the data, left will be no larger than,right will be larger than'''                                               
    # use assert to make sure no parameter can be "None"
    assert data is not None
    assert n_socket is not None
    assert proc_count is not None
    assert MAX_PROCESSES_COUNT is not None

    if (len(data) > 0):
        if (proc_count >= MAX_PROCESSES_COUNT):
            quick(data,0,len(data))
            n_socket.send(data)
            n_socket.close()
        else:
            pivot = data[random.randint(0,len(data)-1)]
            l_data,m_data,r_data = partition(data, pivot)
            recv_left_proc, send_left_proc = multiprocessing.Pipe(duplex=False)
            recv_right_proc, send_right_proc = multiprocessing.Pipe(duplex=False)
            new_proc_count = 2 * proc_count + 1
            left_proc = multiprocessing.Process(target=parallel_quicksort,  \
                                                            args=(l_data,  \
                                                                  send_left_proc,  \
                                                                  new_proc_count,  \
                                                                  MAX_PROCESSES_COUNT))
            right_proc = multiprocessing.Process(target=parallel_quicksort,  \
                                                            args=(r_data,  \
                                                                  send_right_proc,  \
                                                                  new_proc_count,  \
                                                                  MAX_PROCESSES_COUNT))
            left_proc.start()
            right_proc.start()

            l_data = recv_left_proc.recv()
            r_data = recv_right_proc.recv()

            data = np.concatenate((l_data,m_data,r_data))

            n_socket.send(data)
            n_socket.close()
            left_proc.join()
            right_proc.join()
            left_proc.close()
            right_proc.close()
    else:
        n_socket.send(data)
        n_socket.close()

def para_quick_sort(data_in:np.ndarray)->np.ndarray:
    '''use threads to quick sort the data'''
    data = list(data_in)
    recv_socket, send_socket = multiprocessing.Pipe(duplex=False)
    parent_process = multiprocessing.Process(target=parallel_quicksort,  \
                                                            args=(data,  \
                                                                  send_socket, \
                                                                  1,  \
                                                                  multiprocessing.cpu_count()))
    parent_process.start()
    ans = recv_socket.recv()
    parent_process.join()
    return np.array(ans)

@timing
def para_enum_sort(data_in:np.ndarray)->np.ndarray:
    '''parallel enum sort'''
    CL_CODE = '''
    kernel void enumerate(int size, global long* data, global long* buffer) {
        // get group id,when single thread per group
        //const int gid = get_global_id(0);

        // get id when multiple threads in multiple group
        int num_wrk_items  = get_local_size(0);                 
        int local_id = get_local_id(0);                   
        int group_id = get_group_id(0);
        const int gid = (group_id * num_wrk_items + local_id); 

        int index = 0;
        for(int i=0; i<size; i++) {
            if(data[i] < data[gid]) {
                index ++;
            }
            else if(data[i] == data[gid] && i < gid) {
                index ++;
            }
        }
        buffer[index] = data[gid];
    }
    '''

    # create context, queue, program
    plat = cl.get_platforms()
    devices = plat[0].get_devices()
    ctx = cl.Context([devices[0]])
    prg = cl.Program(ctx, CL_CODE).build()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # change data to np.int64, create OpenCL Buffer
    data_np = np.int64(data_in)
    buff_np = np.empty_like(data_np).astype(np.int64)

    #create buffer in device
    data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data_np)
    buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=buff_np)

    # figure originate state
    data_len = np.int32(len(data_np))

    # prg.enumerate(queue, (len(data_np),), (1,), data_len, data, buff)

    work_group_size = 128
    global_size = ((len(data_np)),) #not sure but seems to be all
    local_size = ((work_group_size),)
    prg.enumerate(queue, global_size, local_size, data_len, data, buff)

    #copy buff from device to host
    cl.enqueue_copy(queue, data_np, buff)

    queue.finish()
    data.release()
    buff.release()
    return data_np

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

    # name=generate_num("random_50000_300000.txt",50000,300000)
    # run_sort(name,"orderx.txt",quick_sort)
    # run_sort(name,"orderx.txt",merge_sort)
    # run_sort(name,"orderx.txt",para_merge_sort)
    # run_sort(name,"orderx.txt",para_enum_sort)
    # print(timing)
    
    # run_sort("random.txt","order1.txt",quick_sort)
    # run_sort("random.txt","order2.txt",merge_sort)
    # run_sort("random.txt","order3.txt",enum_sort)
    # run_sort("random.txt","order4.txt",para_merge_sort)
    # run_sort("random.txt","order5.txt",para_enum_sort)
    # print(timing)

    n = random.randint(5, 10)
    data = []
    for i in range(n):
        data.append(random.randint(1, 99))
    ans = para_quick_sort(data)
    print(ans)