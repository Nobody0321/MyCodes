def Insertion_Sort(array):
    for j in range(1, len(array)):
        key = j
        for i in range(j - 1, -1, -1):
            if array[i] - array[key]:
                continue
            else:
                array[key], array[i] = array[i], array[key]
                key = i
    return array


if __name__ == "__main__":
    arr = [5,1,4,3,2,6,9,8,7,0]
    print(Insertion_Sort(arr))