# Inheritance

# Single Inheritance (Example 1)
class idProof():

    def __init__(self,id, idNo):
        self.id = id
        self.idNo = idNo


class eligible():

    def __init__(self, id, idNo, name, age):
        self.name = name
        self.age = age
        idProof.__init__(self, id, idNo)
        
    def confirmation(self):

        if self.age > 18:
            print(f"He's eligible to cast the vote because his age is {self.age}")
            print(f"His details are Name: {self.name}, Age: {self.age}, ID: {self.id}, IdNo: {self.idNo}")
        else:
            print(f"He's not eligibe to cast the vote because his age is {self.age}") 
            print(f"His details are Name: {self.name}, Age: {self.age}, ID: {self.id}, IdNo: {self.idNo}")

election = eligible("Aadhar", 787, "John", 14)
#election.confirmation()


# Single Inheritance (Example 2)

class cars():
    def __init__(self, brand, type):
        self.brand = brand
        self.type = type
        

class exterior():
    def __init__(self, brand, type, color, model):
        self.color = color
        self.model = model
        cars.__init__(self, brand, type)

    def delivery(self):
        print(f"Brand: {self.brand}, Type: {self.type}, Color: {self.color}, Model:{self.model}")


newCar = exterior("Tesla", "Electric", "Yellow", "Sedan")
newCar.delivery()



