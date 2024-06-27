import os
import matplotlib.pyplot as plt

class Node:
    def __init__(self, value, color='RED'):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.color = color


class RedBlackTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.search(value) is not None:
            print("Элемент с таким значением уже существует.")
            return
        else:
            print("Элемент успешно вставлен.")
        node = Node(value)
        if self.root is None:
            self.root = node
            self.root.color = 'BLACK'
        else:
            self._insert_recursive(self.root, node)
            self.fix_violation(node)

    def _insert_recursive(self, root, node):
        if root.value < node.value:
            if root.right is None:
                root.right = node
                node.parent = root
            else:
                self._insert_recursive(root.right, node)
        else:
            if root.left is None:
                root.left = node
                node.parent = root
            else:
                self._insert_recursive(root.left, node)

    def fix_violation(self, node):
        while node != self.root and node.parent.color == 'RED':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle is not None and uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self.left_rotate(node)
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self.right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle is not None and uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self.right_rotate(node)
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self.left_rotate(node.parent.parent)
        self.root.color = 'BLACK'

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def search(self, value):
        return self._search_recursive(self.root, value)

    def _search_recursive(self, root, value):
        if root is None or root.value == value:
            return root
        if root.value < value:
            return self._search_recursive(root.right, value)
        return self._search_recursive(root.left, value)

    def delete(self, value):
        node = self.search(value)
        if node is None:
            print("Такого элемента нет в дереве.")
            return
        else:
            print("Элемент успешно удален.")
        self._delete_node(node)

    def _delete_node(self, node):
        if node.left is None or node.right is None:
            y = node
        else:
            y = self._tree_successor(node)
        if y.left is not None:
            x = y.left
        else:
            x = y.right
        if x is not None:
            x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        if y != node:
            node.value = y.value
        if y.color == 'BLACK':
            self.fix_double_black(x, y.parent)

    def fix_double_black(self, x, parent):
        while x != self.root and (x is None or x.color == 'BLACK'):
            if x == parent.left:
                sibling = parent.right
                if sibling.color == 'RED':
                    sibling.color = 'BLACK'
                    parent.color = 'RED'
                    self.left_rotate(parent)
                    sibling = parent.right
                if (sibling.left is None or sibling.left.color == 'BLACK') and (sibling.right is None or sibling.right.color == 'BLACK'):
                    sibling.color = 'RED'
                    x = parent
                    parent = x.parent
                else:
                    if sibling.right is None or sibling.right.color == 'BLACK':
                        sibling.left.color = 'BLACK'
                        sibling.color = 'RED'
                        self.right_rotate(sibling)
                        sibling = parent.right
                    sibling.color = parent.color
                    parent.color = 'BLACK'
                    sibling.right.color = 'BLACK'
                    self.left_rotate(parent)
                    x = self.root
            else:
                sibling = parent.left
                if sibling.color == 'RED':
                    sibling.color = 'BLACK'
                    parent.color = 'RED'
                    self.right_rotate(parent)
                    sibling = parent.left
                if (sibling.left is None or sibling.left.color == 'BLACK') and (sibling.right is None or sibling.right.color == 'BLACK'):
                    sibling.color = 'RED'
                    x = parent
                    parent = x.parent
                else:
                    if sibling.left is None or sibling.left.color == 'BLACK':
                        sibling.right.color = 'BLACK'
                        sibling.color = 'RED'
                        self.left_rotate(sibling)
                        sibling = parent.left
                    sibling.color = parent.color
                    parent.color = 'BLACK'
                    sibling.left.color = 'BLACK'
                    self.right_rotate(parent)
                    x = self.root
        if x is not None:
            x.color = 'BLACK'

    def _tree_successor(self, node):
        if node.right is not None:
            return self._tree_minimum(node.right)
        parent = node.parent
        while parent is not None and node == parent.right:
            node = parent
            parent = parent.parent
        return parent

    def _tree_minimum(self, node):
        while node.left is not None:
            node = node.left
        return node

    def display(self):
        print("Дерево:")
        self._display_console(self.root)
        print("\nВизуализация дерева:")
        self._display_recursive(self.root, 0, 0)
        plt.show()

    def _display_console(self, root, level=0):
        if root is not None:
            self._display_console(root.right, level + 1)
            print(' ' * 4 * level + '->', root.value, ('R' if root.color == 'RED' else 'B'))
            self._display_console(root.left, level + 1)

    def _display_recursive(self, root, x, y, spacing=20):
        if root is not None:
            plt.text(x, y, str(root.value), color='r' if root.color == 'RED' else 'b', fontsize=12, ha='center')
            if root.left is not None:
                plt.plot([x, x - spacing], [y - 1, y - spacing], color='k')
                self._display_recursive(root.left, x - spacing, y - spacing, spacing / 2)
            if root.right is not None:
                plt.plot([x, x + spacing], [y - 1, y - spacing], color='k')
                self._display_recursive(root.right, x + spacing, y - spacing, spacing / 2)


def print_menu():
    print("|------------------------------|")
    print("|1. Вставить значение в дерево |")
    print("|2. Удалить значение из дерева |")
    print("|3. Визуализировать дерево     |")
    print("|4. Выход                      |")
    print("|------------------------------|")

def main():
    tree = RedBlackTree()
    while True:
        print_menu()
        choice = input("Выберите операцию: ")
        if choice == '1':
            value = int(input("Введите значение для вставки: "))
            tree.insert(value)
            
        elif choice == '2':
            value = int(input("Введите значение для удаления: "))
            tree.delete(value)
        elif choice == '3':
            print("Визуализация дерева:")
            plt.figure()
            plt.axis('off')
            tree.display()
            plt.show()
        elif choice == '4':
            print("Выход.")
            return
        else:
            print("Некорректный выбор операции. Пожалуйста, попробуйте снова.")

if __name__ == "__main__":
    main()
