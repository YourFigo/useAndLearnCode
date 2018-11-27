'''
Created on Nov 27, 2018
@author: Figo
'''
class LNode:
    # 原书中用的__new__()
    # __new__()至少要有一个参数cls，代表当前类，此参数在实例化时由Python解释器自动识别
    # __new__()必须要有返回值，返回实例化出来的实例
    # __init__()有一个参数self，就是这个__new__()返回的实例,__new__负责创建,__init__负责初始化
    # __new__方法主要是当你继承一些不可变的class时(比如int, str, tuple),提供给你一个自定义这些类的实例化过程的途径。
    # 而__init__()无法自定义所要继承的这些不可变类的实例化过程
    def __init__(self,x):
        self.data = x
        self.next = None

def Reverse(head):
    # 判空,我这里多加了一个判断，考虑了只有一个元素节点的情况
    # if head == None or head.next == None:
    if head == None or head.next == None or head.next.next == None:
        return
    # 节点初始化
    pre = None
    cur = None
    next = None
    # 指定第一个节点
    cur = head.next
    next = cur.next
    # 把第一个节点变为尾节点
    cur.next = None
    # 从第二个节点开始循环
    pre = cur
    cur = next
    # 假如只有一个元素节点的情况,原书会出错.
    # 因为假如只有一个节点，next为None，而cur=next,此时cur也为None,而None是没有None.next的
    while cur.next != None:
        next = cur.next
        # 实现逆序
        cur.next = pre
        pre = cur
        cur = next
    # 将最后一个节点指向倒数第二个节点
    cur.next = pre
    # 将head指向最后一个节点
    head.next = cur

if __name__ == '__main__':
    i = 1
    head = LNode(None)
    # print(head)
    # print(type(head))
    # print(head.data)
    # print(head.next)
    # 构造单链表
    tmp = None
    cur = head
    while i < 11:
        tmp = LNode(i)
        # 修改上一次循环的cur.next
        cur.next = tmp
        cur = tmp
        i = i + 1
    print('01_reverseList_1.py')
    print('逆序前：')
    cur = head.next
    while cur != None:
        print(cur.data,end=' ')
        cur = cur.next
    print('\n逆序后：')
    Reverse(head)
    cur = head.next
    while cur != None:
        print(cur.data,end=' ')
        cur = cur.next