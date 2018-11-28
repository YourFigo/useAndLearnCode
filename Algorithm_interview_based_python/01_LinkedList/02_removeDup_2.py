'''
Created on Nov 27, 2018
@author: Figo
'''
class LNode:
    def __init__(self,x):
        self.data = x
        self.next = None
# 将无序链表中的重复元素删除
# 这种方法是 递归删除方法
# 对不含头结点的链表删除重复元素
def removeDupRecursion(first):
    if first is None or first.next is None:
        return first
    pointer = None
    pointerPre = first
    # 对于一个节点，先递归地删除它后面的子链中的所有重复节点，然后找出子链中与该节点相同的节点并删除
    first.next = removeDupRecursion(first.next)
    pointer = first.next
    # 找子链中与该节点重复的元素
    # print('\n')
    while pointer is not None:
        # print(pointer.data,end=' ')
        if first.data == pointer.data:
            pointerPre.next = pointer.next
            pointer = pointerPre.next
        else:
            pointer = pointer.next
            pointerPre = pointerPre.next
    return first
# 对含头结点的链表删除重复元素
def removeDup(head):
    if head is None:
        return
    else:
        head.next = removeDupRecursion(head.next)

if __name__ == '__main__':
    i = 1
    head = LNode(None)
    tmp = None
    cur = head
    while i < 11:
        if i % 2 == 0:
            tmp = LNode(i + 1)
        elif i % 3 == 0:
            tmp = LNode(i - 2)
        else:
            tmp = LNode(2)
        # 尾插法，插入的元素是顺序的；头插法，插入的元素是逆序的
        cur.next = tmp
        cur = cur.next
        i = i + 1
    print('02_removeDup_2.py')
    print('删除重复元素前')
    cur = head.next
    while cur is not None:
        print(cur.data,end=' ')
        cur = cur.next
    print('\n删除重复元素后')
    removeDup(head)
    cur = head.next
    while cur is not None:
        print(cur.data,end=' ')
        cur = cur.next