import random
import multiprocessing
import io
import numpy as np
import pyopencl as cl
from timing import Timing   #import Timing to calculate time

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

def quick_sort(data:np.ndarray)->np.ndarray:
    '''serially quick sort data'''
    quick(data,0,len(data))
    return data

def enum_sort(data:np.ndarray)->np.ndarray:
    '''serial enum sort,????????????????????????????????????????????????'''
    #????????????????????????????????????index???????????????????????????????????????????????????
    ans = np.zeros(len(data),dtype=np.int64)
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

def para_merge_sort(data_in:np.ndarray)->np.ndarray:
    '''parallel compute merge sort'''
    # OpenCL kernel ???????????????
    CL_CODE = '''
    kernel void merge(int chunk_size, int size, global long* data, global long* buff) {
        // ??????????????????
        const int gid = get_global_id(0);

        // ????????????????????????????????????
        const int offset = gid * chunk_size;
        const int real_size = min(offset + chunk_size, size) - offset;
        global long* data_part = data + offset;
        global long* buff_part = buff + offset;

        // ??????????????????????????????
        int r_beg = chunk_size >> 1;
        int b_ptr = 0;
        int l_ptr = 0;
        int r_ptr = r_beg;

        // ????????????
        while (b_ptr < real_size) {
            if (r_ptr >= real_size) {
                // ??????????????????????????????????????????????????????
                buff_part[b_ptr] = data_part[l_ptr++];
            } else if (l_ptr == r_beg) {
                // ??????????????????????????????????????????????????????
                buff_part[b_ptr] = data_part[r_ptr++];
            } else {
                // ??????????????????????????????????????????????????????
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

    # ??????????????? numpy ????????????????????? OpenCL Buffer
    data_np = np.int64(data_in)
    buff_np = np.empty_like(data_np).astype(np.int64)

    # ????????????????????????????????????????????????
    data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data_np)
    buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=buff_np)

    # ???????????????????????????
    data_len = np.int32(len(data_np))
    chunk_size = np.int32(1)

    while chunk_size < data_len:
        # ??????????????????????????????????????????
        chunk_size <<= 1
        # ???????????????????????? 
        group_size = ((data_len - 1) // chunk_size) + 1
        # ????????????????????????
        prg.merge(queue, (group_size,), (1,), chunk_size, data_len, data, buff)
        # ????????????????????????????????????????????????
        temp = data
        data = buff
        buff = temp
        # ?????????????????????
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

    l_data = data[data<pivot].copy()
    m_data = data[data==pivot].copy()
    r_data = data[data>pivot].copy()

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
    recv_socket, send_socket = multiprocessing.Pipe(duplex=False)
    parent_process = multiprocessing.Process(target=parallel_quicksort,  \
                                                            args=(data_in,  \
                                                                  send_socket, \
                                                                  1,  \
                                                                  multiprocessing.cpu_count()))
    parent_process.start()
    ans = recv_socket.recv()
    parent_process.join()
    return np.array(ans)

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

    work_group_size = 64
    global_size = ((len(data_np)),) #not sure but seems to be all
    local_size = ((work_group_size),)
    prg.enumerate(queue, global_size, local_size, data_len, data, buff)

    #copy buff from device to host
    cl.enqueue_copy(queue, data_np, buff)

    queue.finish()
    data.release()
    buff.release()
    return data_np