
def reverse_a_list():
    my_list = [1,2,3,566,6,7,8]
    original_list = my_list.copy()

    list_length = 0
    for i in my_list:
        list_length += 1


    for i in range(int(list_length/2)):

        main_value = my_list[i]
        mirror_value = my_list[list_length-i-1]

        my_list[i] = mirror_value
        my_list[list_length-i-1] = main_value

    print(f"Before reversing {original_list} After revesring {my_list}")

def remove_multiple():
    # Python program to remove multiple
    # elements from a list

    # creating a list
    list1 = [11, 5, 17, 18, 23, 50]

    # Iterate each element in list
    # and add them in variale total
    for ele in list1:
        if ele % 2 == 0:
            list1.remove(ele)

    # printing modified list
    print("New list after removing all even numbers: ", list1)

def count_occurence():
    # Python code to count the number of occurrences
    def countX(lst, x):
        count = 0
        for ele in lst:
            if (ele == x):
                count = count + 1
        return count

    # Driver Code
    lst = [8, 6, 8, 10, 10, 8, 20, 10, 8, 8]
    x = 8
    print('{} has occurred {} times'.format(x, countX(lst, x)))


if __name__ == "__main__":
    # reverse_a_list()
    # remove_multiple()
    count_occurence()