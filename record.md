### 一些注意事项
1. 那个计时我放在runsort里面了，可能时间多了一些读取的应该没关系吧（吧）
   a我又调回来了还是放在各个功能比较好

2.在Google colab上想测性能的时候遇到issue：
    光用cpu的时候：没有设备
    用GPU：invalid arg，让我对硬件兼容性有很大的感触

3.实现快速排序的时候：
    opencl保证cecw？
    没有找到一个确定的答案
    quick sort 问题：放在gpu上不同的责任范围要确定
    换了种写法，那个PRAM-CRCW各个pocessor之间有非常强的依赖性，我个人认为有些是需要同步的。但是不同grid之间理论上不希望有依赖性，所有只能把grid设为1，thread设为size。我在思考这样和thread谁快
    barrier in branch? what will happen? it will hang kernel and crash
    书上的办法不好，太多需要atomic的部分了,可以讲讲我做的尝试：试图还原但是有很大的问题

4.实现enum并行排序的时候
    还是来一个buffer，不要直接在data上面改，太危险了
    有30000个group每个group1thread，和3000000/128个group每个group128个thread性能差距好大耶

take notes: get global id, local id,group id 
https://jorudolph.wordpress.com/2012/02/03/opencl-work-item-ids-globalgrouplocal/

about barrier in cl:
https://stackoverflow.com/questions/6890302/barriers-in-opencl