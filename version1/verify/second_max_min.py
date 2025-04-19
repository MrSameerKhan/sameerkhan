

my_list = [1,2,3,4,5]

def second_large_number ():
    first_largest = my_list[0]
    second_largest = None
    for elem in range(1, len(my_list)):

        if my_list[elem] > first_largest:
            first_largest = my_list[elem]

    print(first_largest, second_largest)



def second():
    data = [-11, 20, 8, 30, 12]

    largest = None
    second_largest = None

    for a in data:

        if not largest or a > largest:
            print(f"largest {largest} and a {a}")
            if largest:
                print(f"if condition largest {largest}")
                second_largest = largest
            largest = a

    print("largest: {}".format(largest))
    print("second_largest: {}".format(second_largest))

def my_second():
    data = [11, 22, 1, 2, 5, 67, 21, 32]

    max1 = data[0]  # largest num
    max2 = data[1]  # second largest num

    for num in data:

        if num > max1:
            print(f"if   Num:{num}  > Max1:{max1}")
            max2 = max1  # Now this number would be second largest
            max1 = num  # This num is largest number in list now.

        # Check with second largest
        elif num > max2:
            print(f"elif Num:{num}  > Max2:{max2}")
            max2 = num  # Now this would be second largest.
        else:
            print(f"else Num:{num}  < Max1:{max1} Max2:{max2} ")


# if __name__ == "__main__":
    # second_large_number()
    # second()
    # my_second()
    
    
    










def sam_large():
    large_list = [5,56,85,6265,7,2316]


    large_num , large_index = large_list[0], 0
    for i in range(1, len(large_list)):

        if large_list[i] > large_num:
            large_num = large_list[i]
            large_index = i

    print(f"The largest number in the given list {large_list} is {large_num} and index is {large_index}")




def sam_small ():

    small_list = [80,18,20,88,22,33]

    small_number, small_index = small_list[0], 0

    for ele in range(1, len(small_list)):

        if small_list[ele] < small_number:
            small_number = small_list[ele]
            small_index = ele

    print(f"The smallest number in the given list {small_list} is {small_number} and index is {small_index}")



def sam_second_largest():
    second_list = [11,22,7,3332,945, 10000]

    first_large_number = second_list[0]
    second_large_number = second_list[1]

    for i in range(len(second_list)):

        if second_list[i] > first_large_number:
            second_large_number = first_large_number
            first_large_number = second_list[i]
            pass
        elif second_list[i] > second_large_number:
            second_large_number = second_list[i]
            pass

    print(f"First large number is {first_large_number} and second large number is {second_large_number}")




def sam_second_smallest():

    second_small_list = [95,5612,641,514,366,23]

    first_small_number = second_small_list[0]
    second_small_number = second_small_list[1]
    third_small_number = second_small_list[2]
    forth_small_number = second_small_list[3]

    for i in range(len(second_small_list)):

        if second_small_list[i] < first_small_number:
            forth_small_number = third_small_number
            third_small_number = second_small_number
            second_small_number = first_small_number
            first_small_number = second_small_list[i]

        elif second_small_list[i] < second_small_number:
            forth_small_number = third_small_number
            third_small_number = second_small_number
            second_small_number = second_small_list[i]

        elif second_small_list[i] < third_small_number:
            forth_small_number = third_small_number
            third_small_number = second_small_list[i]

        elif second_small_list[i] < forth_small_number:
            forth_small_number = second_small_list[i]


    print(f"First {first_small_number}, Second {second_small_number}, Third {third_small_number}, Forth {forth_small_number}")

def clearList():
    # Python3 code to demonstrate
    # clearing a list using
    # *= 0 method

    # Initializing lists
    list1 = [1, 2, 3]

    # Printing list1 before deleting
    print("List1 before deleting is : " + str(list1))

    # deleting list using *= 0
    list1 = list1*0

    # Printing list1 after *= 0
    print("List1 after clearing using *= 0: " + str(list1))


if __name__ == '__main__':
    # sam_large()
    # sam_small()
    # sam_second_largest()
    # sam_second_smallest()
    clearList()
    