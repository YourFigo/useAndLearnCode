'''
Created on Nov 27, 2018
@author: Figo
'''
class LNode:
    def __init__(self,x):
        self.data = x
        self.next = None

# 递归逆向输出链表
def ReversePrint(first):
    if first is None:
        return
    else:
        ReversePrint(first.next)
        print(first.data,end=' ')

if __name__ == '__main__':
    i = 1
    head = LNode(None)
    tmp = None
    cur = head
    while i < 11:
        tmp = LNode(i)
        # 修改上一次循环的cur.next
        cur.next = tmp
        cur = tmp
        i = i + 1

    print('01_reversePrint.py')
    print('正向输出：')
    cur = head.next
    while cur != None:
        print(cur.data, end=' ')
        cur = cur.next

    print('\n逆向输出：')
    ReversePrint(head.next)