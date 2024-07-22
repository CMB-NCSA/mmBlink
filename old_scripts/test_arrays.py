#!/usr/bin/env python
import numpy as np


def stack_cols_lists(c1, c2, append=False):

    # Make sure that c1 and c2 are lists, if not we re-cast them as lists
    if not isinstance(c1, list):
        try:
            c1 = list(c1)
        except Exception as err:
            raise Exception(f"Cannot cast c1 as list {err=}, {type(err)=}")
    if not isinstance(c2, list):
        try:
            c2 = list(c2)
        except Exception as err:
            raise Exception(f"Cannot cast c2 as list {err=}, {type(err)=}")

    # Case 1, we stack c1, c2
    if append is False:

        # Make sure that c1 and c2 have the same dimensions
        if len(c1) != len(c1):
            raise Exception("c1 and c2 have different dimensions")
        else:
            # We will store the new list here as "newlist"
            newlist = []
            for i in range(len(c1)):
                # if elements of c1 are list, we append instead of join
                if isinstance(c1[i], list):
                    c1[i].append(c2[i])
                    x = c1[i]
                # if not we join elements of c1 and c2
                else:
                    x = [c1[i], c2[i]]
                newlist.append(x)

    # Case 2, we want to append c2 to existing c1
    else:
        for c in c2:
            c1.append([c])
        newlist = c1
    return newlist


data = np.zeros((10), dtype=np.ndarray)
c1 = np.array([1, 2, 3, 4, 5])
c2 = np.array([10, 20, 30, 40, 50])
c3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
cx = np.array([-6, -7, -8, -9, -10])
cx2 = np.array([100, 200, 300])

xx = stack_cols_lists(c1, c2)
print("c1 + c2")
print(xx)

# Expand with cx
print("c1 + cx")
xx = stack_cols_lists(xx, cx, append=True)
print(xx)

# Now add a third column
print("xx + c3")
xx = stack_cols_lists(xx, c3)
print(xx)

print("xx + cx2")
xx = stack_cols_lists(xx, cx2, append=True)
print(xx)

u = [np.array(x).mean() for x in xx]
print(u)
exit()


# Add new elements to list as arrays
[xx.append([c]) for c in cx]
print(xx)

# Now append a new column
a = [[xx[i].append(c3[i])] for i in range(len(c3))]
#a = list(np.column_stack((xx, c3)))a
print(xx)
exit()

l = [list(x) for x in list(s)]
[l.append([x]) for x in cx]
print(l)
exit()


[l[i].append(c3[i]) for i in range(len(l))]


print(l)
exit()

print(s)
print(s.shape)
cx = [-6, -7, -8, -9, -10]
a = np.append(s, [cx,])
print(a)

data[0:5] = s # list(s)
data[5:] = [-6, -7, -8, -9, -10]
print(data)

new_data = np.zeros((10), dtype=np.ndarray)
new_data = np.column_stack((data, c3))
new_data = np.column_stack((new_data, c3*2))
print(new_data)
