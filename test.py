from collections import deque

dq = deque(maxlen=3)

dq.append(1)
dq.append(2)
dq.append(3)
print(dq)        # deque([1, 2, 3], maxlen=3)

dq.append(4)
print(dq)        # deque([2, 3, 4], maxlen=3)
