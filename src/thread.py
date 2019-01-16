#--*--coding:utf-8 --*--
'''
Created on 2019年1月10日

@author: adp
'''
import threading
import time
from certifi import __main__
def run():
    time.sleep(2)
    print(threading.current_thread().name)
    time.sleep(2)
    
if __name__=='__main__':
    
    start_time = time.time();
    print('这是主线程：',threading.current_thread().name)
    thread_list = []
    for i in range(5):
        t = threading.Thread(target=run)
        thread_list.append(t)
    
    for t in thread_list:
        t.setDaemon(True)#设置守护线程
        t.start();
        
    for t in thread_list:
        t.join();
    print('主线程结束！',threading.current_thread().name)
    print('一共用时：',time.time()-start_time)
                           