"""
Encapsulation:
The process of wrapping up variables and methods into a single entity is known as Encapsulation.
It acts as a protective shield that puts restrictions on accessing variables and methods directly
"""

"""
Access modifiers:
Access modifiers limit access to the variables and functions of a class. Python uses three types of access modifiers; they are - private, public and protected.
"""
for i in range(1,2):
    print("\n")

# Public members
# Public members are accessible anywhere from the class. All the member variables of the class are by default public.

class iPublic():
    def __init__(self, name):
        self.name = name

sPublic = iPublic("Sameer")
print("Public Name: ", sPublic.name)
 

for i in range(1,2):
    print("\n")













# Protected members
# Protected members are accessible within the class and also available to its sub-classes. To define a protected member, prefix the member name with a single underscore “_”.

class iProtected():
    def __init__(self, name, age):
        self._name = name
        self._age = age
        
sProtected = iProtected("Sameer", 26)
print(f"Protected Name Outside class: {sProtected._name}, Protected Age Outside class: {sProtected._age}")

class derivedProtected():
    def __init__(self, name , age):
        iProtected.__init__(self,name, age)
    
    def printProtectedFromBase(self):
        print(f"Protected Name from Derived Class: {self._name}, Protected Age from Derived Class: {self._age}")

iDerivedProtected = derivedProtected("Shahrukh", 55)
iDerivedProtected.printProtectedFromBase()

for i in range(1,2):
    print("\n")


















# Private members
# Private members are accessible within the class. To define a private member, prefix the member name with a double underscore “__”.


class iPrivate():
    def __init__(self, name, age):
        self.__name = name
        self.__age = age
    def accessPrivate(self):
        print(f"Private Name in same class diff fuction: {self.__name}, Private Age in same class diff function: {self.__age}")
        
sPrivate = iPrivate("Sameer", 26)
sPrivate.accessPrivate()
print(f"Private Name outside class: {sPrivate.__name}, Private Age outside class: {sPrivate.__age}")


class derivedPrivate():
    def __init__(self, name , age):
        iPrivate.__init__(self,name, age)
    
    def printPrivateFromBase(self):
        print(f"Private Name from Derived class: {self.__name}, Private Age from Derived Class: {self.__age}")

iDerivedPrivate = derivedPrivate("Shahrukh", 55)
iDerivedPrivate.printPrivateFromBase()