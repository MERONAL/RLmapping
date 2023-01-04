from collections import Counter
from std_msgs.msg import Int8MultiArray
a = Int8MultiArray()
a = [1,2,3,4,5,1,1,1,1,1,2,2,3]
print(type(a))
m = Counter(a)
print(m)
m[1] = 10086
print(m)