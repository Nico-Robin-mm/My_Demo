import random
from typing import List
import time


def func_test(f: callable) -> bool:
    """
    generate一批随机数组，并检验排序function是否正确
    res: True表示排序函数正确，False表示排序函数错误
    """
    start = time.time()
    
    iters = 30  # 循环 iters 次
    nums_length = 500  # 每次循环的数组长度
    max_num = 30 # 数组从 [0, max_num] 中随机取整数值
    
    res = False
    for i in range(iters):
        nums = [random.randint(0, max_num) for _ in range(nums_length)]
        ret = f(nums)
        for j in range(nums_length - 1):
            if ret[j] > ret[j+1]:
                return res, time.time()-start
    
    res = True
    end = time.time()

    return res, end-start


def select_sort(nums: List[int]) -> List[int]:
    """选择排序"""
    length = len(nums)
    for i in range(length-1):
        min_idx = i
        for j in range(i+1, length):
            if nums[j] < nums[min_idx]:
                min_idx = j
        nums[i], nums[min_idx] = nums[min_idx], nums[i]
    return nums


def insert_sort(nums: List[int]) -> List[int]:
    """插入排序"""
    length = len(nums)
    for i in range(length-1):
        j = i + 1
        temp = nums[j]
        while j > 0 and nums[j-1] > temp:
            nums[j] = nums[j-1]
            j -= 1
        nums[j] = temp
    return nums
        

def merge_sort(nums: List[int]) -> List[int]:
    """归并排序"""
    _merge_helper(nums, 0, len(nums)-1)
    return nums
    
def _merge_helper(nums, first, last):
    if first < last:
        mid = (first+last) // 2
        _merge_helper(nums, first, mid)
        _merge_helper(nums, mid+1, last)
        _merge(nums, first, mid, last)

def _merge(nums, first, mid, last):
    # left [first, ..., mid], right [mid+1, ..., last]
    left = nums[first: mid+1]
    right = nums[mid+1: last+1]
    i, j, k = 0, 0, first
    while i < mid - first + 1 and j < last - mid:
        if left[i] <= right[j]:
            nums[k] = left[i]
            i += 1
            # k += 1
        else:
            nums[k] = right[j]
            j += 1
            # k += 1
        k += 1
    while i < mid - first + 1:
        nums[k] = left[i]
        i += 1
        k += 1
    while j < last - mid:
        nums[k] = right[j]
        j += 1
        k += 1


def quick_sort(nums: List[int]) -> List[int]:
    """快速排序"""
    _quick_helper(nums, 0, len(nums)-1)
    return nums

def _quick_helper(nums, first, last):
    if first < last:
        splitpoint = partition(nums, first, last)
        _quick_helper(nums, first, splitpoint-1)
        _quick_helper(nums, splitpoint+1, last)

def _partition(nums, first, last):
    """partition 第一种 实现方式"""
    pivot = first
    pivot_value = nums[pivot]
    left = first + 1
    right = last
    done = False
    while not done:
        while left <= right and nums[left] <= pivot_value:
            left += 1
        while left <= right and nums[right] >= pivot_value:
            right -= 1
        if left > right:
            done = True
        else:
            nums[left], nums[right] = nums[right], nums[left]
    nums[pivot], nums[right] = nums[right], nums[pivot]
    return right


def partition(nums: List[int], first: int, last: int):
    """partition 第二种 实现方式"""
    pivot_idx = random.randint(first, last)
    nums[first], nums[pivot_idx] = nums[pivot_idx], nums[first]
    pivot = first
    pivot_val = nums[first]
    s = first # s represents splitpoint
    for i in range(first+1, last+1):
        if nums[i] < pivot_val:
            nums[i], nums[s+1] = nums[s+1], nums[i]
            s += 1
    nums[pivot], nums[s] = nums[s], nums[pivot]
    return s


# =============================================================================
def quick_sort_ultimate(nums):  # 对有大量重复值的数组可提高效率
    """平均效率最高的快速排序 三路快排"""
    def helper(nums, first, last):
        if first < last:
            small, equal = __partition(nums, first, last)
            helper(nums, first, small-1)
            helper(nums, equal+1, last)
    helper(nums, 0, len(nums)-1)
    return nums

def __partition(nums: List[int], first: int, last: int):
    # [first, ..., last]
    # random pivot, [pivot, less than pivot, equal to pivot, greater than pivot]
    temp = random.randint(first, last)
    nums[temp], nums[first] = nums[first], nums[temp]
    pivot = first
    pivot_val = nums[pivot]
    pointer = first + 1
    small = first
    equal = first
    big = last
    while pointer <= big:
        if nums[pointer] == pivot_val:
            pointer += 1
            equal += 1
        elif nums[pointer] < pivot_val:
            nums[pointer], nums[small+1] = nums[small+1], nums[pointer]
            pointer += 1
            small += 1
            equal += 1
        else:
            nums[pointer], nums[big] = nums[big], nums[pointer]
            big -= 1
    nums[pivot], nums[small] = nums[small], nums[pivot]
    return small, equal
# =============================================================================

    
def main():
    funcs = [select_sort, insert_sort, merge_sort, quick_sort, quick_sort_ultimate]
    for func in funcs:
        result = func_test(func)
        print(func.__name__ + f"是否准确:{result[0]}, 用时:{result[1]}")



if __name__ == "__main__":
    main()
