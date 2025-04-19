

my_list = [10,20,30,40,50]

def secondLargest(myList):

    largest = 0
    second_largest = 0

    for i in my_list:

        if i > largest:
            print(i , largest)


# Python program to find the largest number
# among the  three numbers using library function

# Driven code
a = 10
b = 14
c = 12
print(max(a, b, c))




if __name__ == '__main__':

    secondLargest(my_list)


