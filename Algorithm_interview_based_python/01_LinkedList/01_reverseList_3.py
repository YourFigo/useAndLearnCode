'''
Created on Nov 27, 2018
@author: Figo
'''
class LNode:
    def __init__(self,x):
        self.data = x
        self.next = None
# 以插入方式实现单链表逆序。
# 每次循环都将当前节点插入到头结点的后面。
# 从第二个节点开始插入，第一个节点直接作为尾节点即可
def Reverse(head):
    if head is None or head.next is None:
        return
    else:
        cur = None
        next = None
        cur = head.next.next
        head.next.next = None
        while cur is not None:
            next = cur.next
            # 先把第一个节点插入到当前节点后面
            cur.next = head.next
            head.next = cur
            cur = next

if __name__ == '__main__':
    i = 1
    head = LNode(None)
    tmp = None
    cur = head
    while i < 11:
        tmp = LNode(i)
        cur.next = tmp
        cur = cur.next
        i = i + 1
    print('01_reverseList_3.py')
    print('逆序前：')
    cur = head.next
    while cur != None:
        print(cur.data,end=' ')
        cur = cur.next
    print('\n逆序后：')
    Reverse(head)
    cur = head.next
    while cur != None:
        print(cur.data, end=' ')
        cur = cur.next

# 三种方法时、空对比
# 三种方法时间复杂度都为o(N),N为链表长度
# 其中 1、就地逆序需要保存前驱节点和后继节点
# 2、递归逆序需要入栈和出栈
# 3、插入逆序相比1不需要保存前驱节点，相比2不需要递归，效率更高。