'''
Created on Nov 27, 2018
@author: Figo
'''
class LNode:
    def __init__(self,x):
        self.data = x
        self.next = None
# 将无序链表中的重复元素删除
# 这种方法是 顺序删除方法，输入带头结点的链表。
# 采用两层循环，外层循环从头到尾遍历链表，内层循环查找外层循环遍历到的节点的所有后继节点，从中找出重复节点然后删除。
def removeDup(head):
    if head is None or head.next is None:
        return
    else:
        outerCur = head.next
        innerCur = None
        innerPre = None
        while outerCur is not None:
            innerPre = outerCur
            innerCur = outerCur.next
            while innerCur is not None:
                # 找到重复并删除
                if outerCur.data == innerCur.data:
                    innerPre.next = innerCur.next
                    innerCur = innerCur.next
                    # 这里innerPre不需要后移，因为删掉重复后，innerPre已经指向innerCur
                else:
                    innerPre = innerCur
                    innerCur = innerCur.next
            outerCur = outerCur.next

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
    print('02_removeDup_1.py')
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

# 顺序删除和递归删除的时间复杂度都为o(N)，但递归的性能要低。
# 顺序删除需要的空间为外层当前节点、内层被删节点和内层被删节点的前驱节点
# 递归删除需要的空间为子链当前节点和子链当前节点的前驱节点