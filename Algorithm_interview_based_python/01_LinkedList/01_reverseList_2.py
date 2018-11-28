'''
Created on Nov 27, 2018
@author: Figo
'''
class LNode:
    def __init__(self,x):
        self.data = x
        self.next = None
# 递归方式对不带头结点的链表进行逆序，输入第一个节点
# 每次递归都实现除当前节点外的所有节点逆序，然后将当前节点放在逆序后的链表的尾部
def RecursiveReverse(first):
    # 如果链表为空或者只有一个节点
    if first is None or first.next is None:
        return first
    else:
        newFirst = RecursiveReverse(first.next)
        # 把当前节点加在后面节点逆序后的链表的尾部
        first.next.next = first
        first.next = None
    return newFirst
# 递归方式对带头结点的链表进行逆序，输入头结点
def Reverse(head):
    if head is None:
        return
    # 获取第一个节点
    first = head.next
    # 逆序后的新的第一个节点
    newFirst = RecursiveReverse(first)
    head.next = newFirst
    # 原书这个地方返回的不太对，他返回的是newFirst
    return head

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
    print('01_reverseList_2.py')
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