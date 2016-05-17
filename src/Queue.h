#ifndef QUEUE_H
#define QUEUE_H

#include <cctype>
#include <iostream>

using namespace std;

template <typename T>
struct Node {
    T element;
    Node *next;
};

template <typename T>
class Queue {
public:
    Queue();
    ~Queue();
    void push(const T &val);
    void pop();
    T front();
    size_t size() const;
private:
    Node<T> *_front, *_tail;
    size_t _size;
};

template <typename T>
Queue<T>::Queue() {
    _front = _tail = NULL;
    _size = 0;
}

template <typename T>
Queue<T>::~Queue() {
    while (_front != NULL) {
        Node<T> *temp = _front;
        _front = _front->next;
        delete temp;
    }
}

template <typename T>
void Queue<T>::push(const T &val) {
    Node<T> *temp = new Node<T>();
    temp->element = val;
    if (_front == NULL) {
        _front = temp;
        _tail = temp;
    } else {
        temp->next = NULL;
        _tail->next = temp;
        _tail = temp;
    }
    _size++;
}

template <typename T>
void Queue<T>::pop() {
    Node<T> *temp = _front;
    _front = _front->next;
    delete temp;
    _size--;
}

template <typename T>
size_t Queue<T>::size() const {
    return _size;
}

template <typename T>
T Queue<T>::front() {
    return _front->element;
}

#endif /* QUEUE_H */

