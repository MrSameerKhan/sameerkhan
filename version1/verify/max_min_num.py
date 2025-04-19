

my_list = [11,25,65,89,760, 3, 1,5, 7,9,11, 89, 1001]

def largest_number():

    lar_num = my_list[0]
    print(f"lar num is {lar_num}")

    for i in range(1, len(my_list)):
        print(f"Index is {i}")
        if my_list[i] > lar_num:
            print(f"  {my_list[i]} > {lar_num}")
            lar_num = my_list[i]
            print(f"Update {lar_num}")
        else:
            print(f"Don't update cause {my_list[i]} < {lar_num} ")




def smallest_number():

    small_number , small_position = my_list[0], 0
    for elem in range(1, len(my_list)):

        if my_list[elem] < small_number:
            small_number = my_list[elem]
            small_position = elem

    print(f"The smallest number in the list in {small_number} and the index of it is {small_position}")



if __name__ == "__main__":
    # largest_number()
    smallest_number()



